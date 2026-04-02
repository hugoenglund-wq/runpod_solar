from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CopiedFile:
    relative_path: str
    bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy trained model artifacts into a git-tracked snapshot folder."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Source artifacts directory, e.g. /workspace/artifacts",
    )
    parser.add_argument(
        "--snapshot-name",
        required=True,
        help="Destination snapshot name, e.g. runpod_2026-04-02_v1",
    )
    parser.add_argument(
        "--dest-root",
        default="published_models",
        help="Git-tracked destination root inside the repo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing snapshot directory if it already exists.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> list[CopiedFile]:
    copied: list[CopiedFile] = []
    if not src.exists():
        return copied

    for file_path in sorted(src.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(src)
        target = dst / relative
        ensure_dir(target.parent)
        shutil.copy2(file_path, target)
        copied.append(
            CopiedFile(
                relative_path=str((dst.name and (Path(dst.name) / relative)) or relative).replace("\\", "/"),
                bytes=file_path.stat().st_size,
            )
        )
    return copied


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(args.artifacts_dir).resolve()
    dest_root = (repo_root / args.dest_root).resolve()
    snapshot_dir = dest_root / args.snapshot_name

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")

    if snapshot_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Snapshot already exists: {snapshot_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(snapshot_dir)

    ensure_dir(snapshot_dir)

    copied_files: list[CopiedFile] = []
    copied_files.extend(copy_tree(artifacts_dir / "models", snapshot_dir / "models"))
    copied_files.extend(copy_tree(artifacts_dir / "metrics", snapshot_dir / "metrics"))

    manifest = {
        "snapshot_name": args.snapshot_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts_dir": str(artifacts_dir),
        "repo_destination": str(snapshot_dir),
        "copied_files": [asdict(item) for item in copied_files],
    }

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Published snapshot to {snapshot_dir}")
    print(f"Copied {len(copied_files)} files from {artifacts_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
