#!/usr/bin/env python3
"""
whisper_transcribe - Transcreve audio em texto usando Whisper.

Suporta duas abordagens:

  * "light"  -> faster-whisper. Reimplementacao otimizada, ideal para rodar
                em CPU (maquina sem GPU dedicada). Mais leve e rapido na CPU.

  * "heavy"  -> openai-whisper. Implementacao de referencia, roda melhor com
                GPU (CUDA). Usa mais memoria; em CPU fica lento.

O Whisper SOMENTE transcreve (audio -> texto). Ele nao gera resumo, ata ou
insights - isso e tarefa de um LLM em uma etapa posterior do fluxo.

Este modulo pode ser usado de forma programatica (funcao transcribe) ou como
biblioteca importada pelo orquestrador (app.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Tamanhos de modelo Whisper, do mais leve/rapido ao mais pesado/preciso.
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]


# --------------------------------------------------------------------------- #
# Deteccao de dependencias e hardware
# --------------------------------------------------------------------------- #
def has_module(name: str) -> bool:
    """Verifica se um modulo Python esta instalado, sem importa-lo de fato."""
    import importlib.util
    return importlib.util.find_spec(name) is not None


# Mapa: abordagem -> (nome do modulo importavel, nome do pacote no pip).
REQUIRED_PACKAGE = {
    "light": ("faster_whisper", "faster-whisper"),
    "heavy": ("whisper", "openai-whisper"),
}

# Permite instalacao automatica sem perguntar (usado pelo orquestrador quando
# o usuario ja confirmou, ou em execucao nao-interativa).
AUTO_INSTALL = False


def ensure_package(module_name: str, pip_name: str) -> None:
    """
    Garante que um pacote esteja instalado. Se faltar, instala sob demanda
    (na primeira vez), pedindo confirmacao ao usuario. Assim so baixamos a
    biblioteca da abordagem realmente escolhida.
    """
    if has_module(module_name):
        return

    import subprocess

    print(f"  A biblioteca '{pip_name}' ainda nao esta instalada.")
    if not AUTO_INSTALL:
        try:
            resp = input(f"  Instalar agora com pip? (pode baixar centenas de MB) [S/n]: ").strip().lower()
        except EOFError:
            resp = "n"
        if resp not in ("", "s", "sim", "y", "yes"):
            raise RuntimeError(
                f"'{pip_name}' e necessario para esta abordagem.\n"
                f"  Instale manualmente com: pip install {pip_name}"
            )

    print(f"  Instalando '{pip_name}'... (isso pode levar alguns minutos)")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name],
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Falha ao instalar '{pip_name}'.\n"
            f"  Tente manualmente: pip install {pip_name}"
        )
    if not has_module(module_name):
        raise RuntimeError(
            f"'{pip_name}' foi instalado mas o modulo '{module_name}' "
            f"ainda nao pode ser importado. Verifique o ambiente Python."
        )
    print(f"  '{pip_name}' instalado com sucesso.")


def gpu_available() -> bool:
    """Retorna True se houver GPU CUDA disponivel (via torch, se instalado)."""
    if not has_module("torch"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Abordagem LEVE: faster-whisper (otimizada para CPU)
# --------------------------------------------------------------------------- #
def transcribe_light(audio_path: Path, model_size: str = "small",
                     language: str | None = None,
                     with_timestamps: bool = False) -> str:
    """
    Transcreve usando faster-whisper. Roda bem em CPU.

    Requer: faster-whisper (instalado sob demanda na primeira execucao).
    """
    ensure_package(*REQUIRED_PACKAGE["light"])
    from faster_whisper import WhisperModel

    # Em CPU usamos compute_type int8 (menor uso de memoria e mais rapido).
    device = "cuda" if gpu_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"  [light] carregando modelo '{model_size}' em {device} "
          f"(compute_type={compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(str(audio_path), language=language)
    print(f"  idioma detectado: {info.language} "
          f"(confianca {info.language_probability:.2f})")

    linhas: list[str] = []
    for seg in segments:
        if with_timestamps:
            linhas.append(f"[{_fmt_ts(seg.start)} -> {_fmt_ts(seg.end)}] {seg.text.strip()}")
        else:
            linhas.append(seg.text.strip())
    return "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Abordagem PESADA: openai-whisper (referencia, melhor com GPU)
# --------------------------------------------------------------------------- #
def transcribe_heavy(audio_path: Path, model_size: str = "large-v3",
                     language: str | None = None,
                     with_timestamps: bool = False) -> str:
    """
    Transcreve usando openai-whisper. Recomendado com GPU (CUDA).

    Requer: openai-whisper (instalado sob demanda na primeira execucao).
    """
    ensure_package(*REQUIRED_PACKAGE["heavy"])
    import whisper

    device = "cuda" if gpu_available() else "cpu"
    if device == "cpu":
        print("  [heavy] AVISO: nenhuma GPU CUDA detectada. Rodando em CPU "
              "(pode ficar lento). Considere a abordagem 'light'.")

    print(f"  [heavy] carregando modelo '{model_size}' em {device}...")
    model = whisper.load_model(model_size, device=device)

    result = model.transcribe(str(audio_path), language=language,
                              fp16=(device == "cuda"))
    print(f"  idioma detectado: {result.get('language', '?')}")

    if with_timestamps:
        linhas = [
            f"[{_fmt_ts(s['start'])} -> {_fmt_ts(s['end'])}] {s['text'].strip()}"
            for s in result.get("segments", [])
        ]
        return "\n".join(linhas)
    return result["text"].strip()


# --------------------------------------------------------------------------- #
# Ponto de entrada unico
# --------------------------------------------------------------------------- #
def transcribe(audio_path: Path, approach: str = "light",
               model_size: str | None = None, language: str | None = None,
               with_timestamps: bool = False) -> str:
    """
    Despacha para a abordagem escolhida.

    approach: "light" (CPU/faster-whisper) ou "heavy" (GPU/openai-whisper).
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio nao encontrado: {audio_path}")

    if approach == "light":
        return transcribe_light(audio_path, model_size or "small",
                                language, with_timestamps)
    if approach == "heavy":
        return transcribe_heavy(audio_path, model_size or "large-v3",
                                language, with_timestamps)
    raise ValueError(f"Abordagem invalida: {approach!r}. Use 'light' ou 'heavy'.")


def _fmt_ts(seconds: float) -> str:
    """Formata segundos como MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Uso direto pela linha de comando
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcreve um audio em texto usando Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="Arquivo de audio (ex.: MP3).")
    parser.add_argument("-a", "--approach", choices=["light", "heavy"],
                        default="light",
                        help="light = CPU (faster-whisper); heavy = GPU (openai-whisper).")
    parser.add_argument("-m", "--model", choices=MODEL_SIZES,
                        help="Tamanho do modelo. Padrao: 'small' (light) ou 'large-v3' (heavy).")
    parser.add_argument("-l", "--language",
                        help="Idioma (ex.: pt). Deixe vazio para deteccao automatica.")
    parser.add_argument("-o", "--output",
                        help="Arquivo .txt de saida (padrao: mesmo nome do audio).")
    parser.add_argument("--timestamps", action="store_true",
                        help="Inclui marcacao de tempo por trecho.")
    args = parser.parse_args()

    audio = Path(args.audio)
    print(f"Transcrevendo: {audio.name} (abordagem: {args.approach})")
    try:
        texto = transcribe(audio, args.approach, args.model, args.language,
                           args.timestamps)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output) if args.output else audio.with_suffix(".txt")
    out.write_text(texto, encoding="utf-8")
    print(f"\nTranscricao salva em: {out}")
    print(f"({len(texto)} caracteres)")


if __name__ == "__main__":
    main()
