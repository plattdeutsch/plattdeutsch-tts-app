from pathlib import Path
import argparse
import sys
import shutil


DEFAULT_TTS_NAME = "tts_models/de/thorsten/tacotron2-DDC"
DEFAULT_VOCODER_NAME = "vocoder_models/de/thorsten/hifigan_v1"


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob('*'):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Coqui TTS model + vocoder into ./models using TTS downloader")
    parser.add_argument("--model", default=DEFAULT_TTS_NAME, help="TTS model name (default: %(default)s)")
    parser.add_argument("--vocoder", default=DEFAULT_VOCODER_NAME, help="Vocoder model name (default: %(default)s)")
    args = parser.parse_args()

    # Use Coqui TTS downloader to fetch to cache, then copy into ./models
    try:
        from TTS.utils.downloaders import download_model
    except Exception as e:
        print("Could not import Coqui downloader from TTS. Is TTS installed?", file=sys.stderr)
        raise

    root = Path(__file__).parent
    models_root = root / "models"

    print(f"Fetching {args.model} via Coqui downloader…")
    tts_src = Path(download_model(args.model))
    tts_dest = models_root / args.model.replace('/', '--')
    print(f"Copying {tts_src} -> {tts_dest}")
    copy_tree(tts_src, tts_dest)

    print(f"Fetching {args.vocoder} via Coqui downloader…")
    voc_src = Path(download_model(args.vocoder))
    voc_dest = models_root / args.vocoder.replace('/', '--')
    print(f"Copying {voc_src} -> {voc_dest}")
    copy_tree(voc_src, voc_dest)

    # Quick presence checks
    tts_ok = (tts_dest / "config.json").exists()
    voc_ok = (voc_dest / "config.json").exists()
    if not tts_ok or not voc_ok:
        print("Warning: Expected config.json not found in one of the destinations.", file=sys.stderr)

    print("Done. The Flask app expects these folders:")
    print(f" - {tts_dest}")
    print(f" - {voc_dest}")


if __name__ == "__main__":
    main()
