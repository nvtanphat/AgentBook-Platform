from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test AgentBook upload API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI base URL")
    parser.add_argument("--file", type=Path, default=None, help="Optional file to upload")
    parser.add_argument("--owner-id", default="user_demo")
    parser.add_argument("--collection-name", default="Machine Learning")
    parser.add_argument("--subject", default="Machine Learning")
    parser.add_argument("--topic", default="Regularization")
    return parser


def default_pdf() -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = Path(handle.name)
    handle.write(b"%PDF-1.4\n% AgentBook smoke test file\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    handle.close()
    return path


def main() -> None:
    args = build_parser().parse_args()
    upload_path = args.file or default_pdf()
    metadata = {
        "owner_id": args.owner_id,
        "collection_name": args.collection_name,
        "subject": args.subject,
        "topic": args.topic,
        "language": "en",
        "modality": "mixed",
        "source_type": "smoke_test",
    }
    with upload_path.open("rb") as file_handle:
        response = httpx.post(
            f"{args.base_url.rstrip('/')}/api/v1/materials/upload",
            data={"metadata": json.dumps(metadata)},
            files={"file": (upload_path.name, file_handle, "application/pdf")},
            timeout=30.0,
        )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
