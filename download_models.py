"""Downloads HazardGuard model files from private HF model repo at runtime."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN secret is not set; cannot download private model repo")

    repo_id = os.environ.get("MODEL_REPO_ID", "projectgaia/ShrishtiAI-models")
    model_root = os.environ.get("MODEL_ROOT_PATH", "/app/models")
    local_dir = Path(model_root)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models from {repo_id} into {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=["hazardguard/**"],
        ignore_patterns=["*.git*", ".gitattributes"],
    )
    print("Models downloaded successfully.")


if __name__ == "__main__":
    main()
