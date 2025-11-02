import shutil
from pathlib import Path
import argparse


DEFAULT_MODEL_SUBDIR = "tts_models--de--thorsten--tacotron2-DDC"
DEFAULT_VOCODER_SUBDIR = "vocoder_models--de--thorsten--hifigan_v1"


def remove_dir(path: Path) -> bool:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def purge_hf_cache(model_subdirs: list[str]) -> None:
    try:
        from huggingface_hub.constants import HF_HOME
    except Exception:
        return

    hf_models_root = Path(HF_HOME) / "hub"
    for sub in model_subdirs:
        # HF cache uses repo_id encoded as models--<org>--<name>
        # Our repos are under org "coqui-ai" and name equals subdir
        cache_dir = hf_models_root / f"models--coqui-ai--{sub}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove local TTS model folders under ./models")
    parser.add_argument(
        "--purge-cache",
        action="store_true",
        help="Also delete matching Hugging Face cache entries for these repos",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    models_root = root / "models"
    targets = [
        models_root / DEFAULT_MODEL_SUBDIR,
        models_root / DEFAULT_VOCODER_SUBDIR,
    ]

    any_removed = False
    for t in targets:
        removed = remove_dir(t)
        print(f"Removed {t}" if removed else f"Not found {t}")
        any_removed = any_removed or removed

    if args.purge_cache:
        purge_hf_cache([DEFAULT_MODEL_SUBDIR, DEFAULT_VOCODER_SUBDIR])
        print("Attempted to purge matching Hugging Face cache entries (if present).")

    if not any_removed:
        print("Nothing to remove under ./models.")


if __name__ == "__main__":
    main()

