"""Prepare the pinned public model without requiring a machine-specific proxy."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "models/qwen2.5-0.5b-instruct"
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
FILES = ["config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "model.safetensors", "LICENSE"]


def main():
    if not all((DEST / name).is_file() for name in FILES):
        snapshot_download(REPO, revision=REVISION, local_dir=DEST,
                          allow_patterns=FILES)
    with (DEST / "model.safetensors").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != WEIGHT_SHA256:
        raise RuntimeError(f"Model weight SHA-256 mismatch: {actual}")
    (DEST / "source.json").write_text(json.dumps(
        dict(repo_id=REPO, revision=REVISION, weight_sha256=actual), indent=2) + "\n")
    print(f"Verified {REPO} at {REVISION}")


if __name__ == "__main__":
    main()
