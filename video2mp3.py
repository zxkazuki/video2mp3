#!/usr/bin/env python3
"""
video2mp3 - Extrai o audio de videos e converte para MP3 otimizado.

O audio e re-encodado de verdade pelo FFmpeg (nao e renomear extensao).
As configuracoes padrao sao otimizadas para transcricao por IA / geracao
de atas: mono, 16 kHz, bitrate baixo -> arquivo muito menor que o video,
mantendo a voz perfeitamente inteligivel.

Uso:
    python video2mp3.py entrada.mp4
    python video2mp3.py uma_pasta/            # converte todos os videos da pasta
    python video2mp3.py entrada.mp4 -o saida.mp3
    python video2mp3.py entrada.mp4 --bitrate 64k --sample-rate 22050
    python video2mp3.py entrada.mp4 --quality music   # estereo, qualidade maior
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Extensoes de video reconhecidas ao processar uma pasta.
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".ts"}

# Perfis prontos: (canais, sample_rate_hz, bitrate)
PROFILES = {
    # Otimizado para transcricao/atas: menor arquivo possivel, voz clara.
    "voice": {"channels": 1, "sample_rate": 16000, "bitrate": "48k"},
    # Meio-termo, um pouco mais de qualidade.
    "balanced": {"channels": 1, "sample_rate": 22050, "bitrate": "64k"},
    # Qualidade de musica: estereo, 44.1 kHz.
    "music": {"channels": 2, "sample_rate": 44100, "bitrate": "192k"},
}


def check_ffmpeg() -> str:
    """Retorna o caminho do ffmpeg ou encerra com instrucoes de instalacao."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    print(
        "ERRO: FFmpeg nao encontrado no PATH.\n\n"
        "Instale com um destes metodos (Windows):\n"
        "  winget install Gyan.FFmpeg\n"
        "  choco install ffmpeg\n"
        "  scoop install ffmpeg\n\n"
        "Depois feche e reabra o terminal e tente novamente.",
        file=sys.stderr,
    )
    sys.exit(1)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def convert(ffmpeg: str, src: Path, dst: Path, channels: int, sample_rate: int,
            bitrate: str, overwrite: bool) -> bool:
    """Converte um unico arquivo. Retorna True em caso de sucesso."""
    if dst.exists() and not overwrite:
        print(f"  [pulado] ja existe: {dst.name} (use --overwrite para sobrescrever)")
        return False

    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i", str(src),
        "-vn",                      # descarta o video, so audio
        "-ac", str(channels),       # numero de canais (1 = mono)
        "-ar", str(sample_rate),    # taxa de amostragem
        "-b:a", bitrate,            # bitrate do audio
        "-codec:a", "libmp3lame",   # encoder MP3
        str(dst),
    ]

    print(f"  convertendo: {src.name} -> {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [erro] falha ao converter {src.name}:", file=sys.stderr)
        print("  " + result.stderr.strip().splitlines()[-1] if result.stderr else "", file=sys.stderr)
        return False

    in_size = src.stat().st_size
    out_size = dst.stat().st_size
    reducao = (1 - out_size / in_size) * 100 if in_size else 0
    print(f"  OK: {human_size(in_size)} -> {human_size(out_size)} "
          f"(reducao de {reducao:.1f}%)")
    return True


def collect_inputs(path: Path) -> list[Path]:
    """Retorna a lista de arquivos de video a processar."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    print(f"ERRO: caminho nao encontrado: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai audio de videos e converte para MP3 otimizado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Arquivo de video ou pasta com videos.")
    parser.add_argument("-o", "--output",
                        help="Arquivo MP3 de saida (apenas quando a entrada e um unico arquivo).")
    parser.add_argument("-d", "--output-dir",
                        help="Pasta de saida para os MP3s (padrao: mesma pasta do video).")
    parser.add_argument("-q", "--quality", choices=PROFILES.keys(), default="voice",
                        help="Perfil de qualidade.")
    parser.add_argument("--channels", type=int,
                        help="Sobrescreve o numero de canais (1 = mono, 2 = estereo).")
    parser.add_argument("--sample-rate", type=int,
                        help="Sobrescreve a taxa de amostragem em Hz.")
    parser.add_argument("--bitrate",
                        help="Sobrescreve o bitrate do audio (ex.: 48k, 64k, 128k).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescreve arquivos MP3 existentes.")
    args = parser.parse_args()

    ffmpeg = check_ffmpeg()

    profile = PROFILES[args.quality].copy()
    if args.channels is not None:
        profile["channels"] = args.channels
    if args.sample_rate is not None:
        profile["sample_rate"] = args.sample_rate
    if args.bitrate is not None:
        profile["bitrate"] = args.bitrate

    input_path = Path(args.input)
    files = collect_inputs(input_path)
    if not files:
        print("Nenhum video encontrado para converter.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Perfil: {args.quality} -> {profile['channels']} canal(is), "
          f"{profile['sample_rate']} Hz, {profile['bitrate']}")
    print(f"{len(files)} arquivo(s) para converter.\n")

    ok = 0
    for src in files:
        if args.output and len(files) == 1:
            dst = Path(args.output)
        else:
            base = output_dir if output_dir else src.parent
            dst = base / (src.stem + ".mp3")
        if convert(ffmpeg, src, dst, profile["channels"], profile["sample_rate"],
                   profile["bitrate"], args.overwrite):
            ok += 1

    print(f"\nConcluido: {ok}/{len(files)} convertido(s) com sucesso.")


if __name__ == "__main__":
    main()
