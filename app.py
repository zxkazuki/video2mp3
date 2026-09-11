#!/usr/bin/env python3
"""
app.py - Orquestrador do fluxo: video -> MP3 -> transcricao (Whisper).

Roda no terminal de forma interativa:
  1. Pergunta o arquivo/pasta de video.
  2. Converte para MP3 otimizado (reaproveitando video2mp3.py).
  3. SEMPRE pergunta qual Whisper usar:
        - leve  (CPU / faster-whisper)
        - pesado (GPU / openai-whisper)
  4. Gera um .txt com a transcricao de cada audio.

A transcricao resultante e o texto bruto do que foi falado. A geracao da ata
/ insights e uma etapa posterior (ex.: sua skill no Amazon Q).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reaproveita a logica ja testada de conversao.
import video2mp3
import whisper_transcribe as wt


# --------------------------------------------------------------------------- #
# Helpers de interacao no terminal
# --------------------------------------------------------------------------- #
def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    resp = input(f"{prompt}{suffix}: ").strip()
    return resp or (default or "")


def ask_choice(prompt: str, options: dict[str, str], default: str) -> str:
    """options: {tecla: descricao}. Retorna a tecla escolhida."""
    print(prompt)
    for key, desc in options.items():
        marca = " (padrao)" if key == default else ""
        print(f"  [{key}] {desc}{marca}")
    while True:
        resp = input("Escolha: ").strip().lower() or default
        if resp in options:
            return resp
        print(f"  Opcao invalida. Use uma de: {', '.join(options)}")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "S/n" if default else "s/N"
    resp = input(f"{prompt} [{d}]: ").strip().lower()
    if not resp:
        return default
    return resp in ("s", "sim", "y", "yes")


# --------------------------------------------------------------------------- #
# Etapas do fluxo
# --------------------------------------------------------------------------- #
def step_convert(ffmpeg: str, videos: list[Path], out_dir: Path | None) -> list[Path]:
    """Converte os videos para MP3 e retorna a lista de MP3s gerados."""
    profile = video2mp3.PROFILES["voice"]
    print(f"\n== Etapa 1: conversao para MP3 ==")
    print(f"Perfil: voice -> {profile['channels']} canal(is), "
          f"{profile['sample_rate']} Hz, {profile['bitrate']}")

    mp3s: list[Path] = []
    for src in videos:
        base = out_dir if out_dir else src.parent
        dst = base / (src.stem + ".mp3")
        if video2mp3.convert(ffmpeg, src, dst, profile["channels"],
                             profile["sample_rate"], profile["bitrate"],
                             overwrite=True):
            mp3s.append(dst)
    return mp3s


def step_transcribe(mp3s: list[Path], approach: str, model: str | None,
                    language: str | None, timestamps: bool,
                    out_dir: Path | None) -> list[Path]:
    """Transcreve cada MP3 e retorna a lista de .txt gerados."""
    print(f"\n== Etapa 2: transcricao (Whisper - {approach}) ==")
    txts: list[Path] = []
    for mp3 in mp3s:
        print(f"\nTranscrevendo: {mp3.name}")
        try:
            texto = wt.transcribe(mp3, approach, model, language, timestamps)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"  ERRO: {e}", file=sys.stderr)
            continue
        base = out_dir if out_dir else mp3.parent
        out = base / (mp3.stem + ".txt")
        out.write_text(texto, encoding="utf-8")
        print(f"  transcricao salva: {out.name} ({len(texto)} caracteres)")
        txts.append(out)
    return txts


# --------------------------------------------------------------------------- #
# Fluxo principal interativo
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 60)
    print(" Video -> MP3 -> Transcricao (Whisper)")
    print("=" * 60)

    # Pre-requisito: FFmpeg.
    ffmpeg = video2mp3.check_ffmpeg()

    # 1. Entrada.
    entrada = ask("\nArquivo de video ou pasta")
    if not entrada:
        print("Nada informado. Saindo.")
        sys.exit(0)
    videos = video2mp3.collect_inputs(Path(entrada))
    if not videos:
        print("Nenhum video encontrado.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(videos)} video(s) encontrado(s).")

    # Pasta de saida (opcional).
    saida = ask("Pasta de saida (vazio = mesma pasta dos videos)", default="")
    out_dir = Path(saida) if saida else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 2. SEMPRE pergunta qual Whisper usar.
    print()
    gpu = wt.gpu_available()
    print(f"GPU CUDA detectada: {'sim' if gpu else 'nao'}")
    approach = ask_choice(
        "\nQual Whisper deseja usar?",
        {
            "1": "Leve  - CPU (faster-whisper). Recomendado sem GPU dedicada.",
            "2": "Pesado - GPU (openai-whisper). Recomendado com GPU CUDA.",
        },
        default="1" if not gpu else "2",
    )
    approach = "light" if approach == "1" else "heavy"

    # Instalacao sob demanda: so baixa a lib da abordagem escolhida, e apenas
    # se ainda nao estiver instalada. Confirma uma unica vez aqui.
    module_name, pip_name = wt.REQUIRED_PACKAGE[approach]
    if not wt.has_module(module_name):
        print(f"\nA abordagem '{approach}' usa '{pip_name}', que ainda nao esta instalado.")
        if ask_yes_no(f"Instalar '{pip_name}' agora na primeira execucao?", default=True):
            wt.AUTO_INSTALL = True  # nao perguntar de novo durante o fluxo
        else:
            print(f"Sem '{pip_name}' nao e possivel transcrever. "
                  f"Instale com: pip install {pip_name}")
            sys.exit(1)

    # Modelo (usa padrao da abordagem se vazio).
    default_model = "small" if approach == "light" else "large-v3"
    modelo = ask(f"Tamanho do modelo {wt.MODEL_SIZES}", default=default_model)

    # Idioma e timestamps.
    idioma = ask("Idioma (ex.: pt; vazio = detectar automaticamente)", default="")
    idioma = idioma or None
    timestamps = ask_yes_no("Incluir marcacao de tempo (timestamps)?", default=False)

    # Executa o fluxo.
    mp3s = step_convert(ffmpeg, videos, out_dir)
    if not mp3s:
        print("Nenhum MP3 gerado. Encerrando.", file=sys.stderr)
        sys.exit(1)

    txts = step_transcribe(mp3s, approach, modelo, idioma, timestamps, out_dir)

    # Resumo final.
    print("\n" + "=" * 60)
    print(f" Concluido: {len(mp3s)} audio(s), {len(txts)} transcricao(oes).")
    print("=" * 60)
    if txts:
        print("Transcricoes geradas:")
        for t in txts:
            print(f"  - {t}")
        print("\nProximo passo: use esses .txt na sua skill do Amazon Q "
              "para gerar a ata / insights.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")
        sys.exit(130)
