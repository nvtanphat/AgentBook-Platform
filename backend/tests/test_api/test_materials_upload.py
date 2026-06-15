from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import BaseModel
from beanie import PydanticObjectId

from src.core.config import get_settings
from src.core.security import UploadValidationError
from src.dependencies import get_material_service
from src.models.common import PipelineStatus
from src.main import app
from src.services.material_service import MaterialService
import src.services.material_service as material_service_module
from src.models.common import utc_now
from src.schemas.material import MaterialUploadResponse


class FakeMaterialService:
    async def upload_material_from_temp(self, **kwargs) -> MaterialUploadResponse:
        metadata = kwargs["metadata"]
        if kwargs["original_filename"].lower().endswith(".pdf") and not kwargs["head"].startswith(b"%PDF"):
            raise UploadValidationError("PDF file must have PDF magic bytes")
        return MaterialUploadResponse(
            material_id="65f000000000000000000001",
            doc_id="65f000000000000000000001",
            collection_id=metadata.collection_id or "65f000000000000000000002",
            job_id="job-123",
            status="uploaded",
            stage="uploaded",
            filename="safe.pdf",
            original_name=kwargs["original_filename"],
            checksum_sha256="a" * 64,
            file_size_bytes=kwargs["file_size_bytes"],
            storage_path="raw/user_demo/65f000000000000000000002/safe.pdf",
        )


class DuplicateMaterialService:
    async def upload_material_from_temp(self, **kwargs) -> MaterialUploadResponse:
        raise ValueError("File already exists in this collection")


def test_upload_route_accepts_valid_pdf(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_material_service] = lambda: FakeMaterialService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/materials/upload",
                data={
                    "metadata": json.dumps(
                        {
                            "owner_id": "user_demo",
                            "collection_name": "Machine Learning",
                            "subject": "Machine Learning",
                            "topic": "Regularization",
                            "language": "en",
                        }
                    )
                },
                files={"file": ("lecture.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
            )
        assert response.status_code == 201
        body = response.json()
        assert body["success"] is True
        assert body["data"]["status"] == "uploaded"
        assert body["data"]["doc_id"] == "65f000000000000000000001"
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_upload_route_rejects_bad_magic_bytes(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_material_service] = lambda: FakeMaterialService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/materials/upload",
                data={"metadata": json.dumps({"owner_id": "user_demo", "collection_name": "Machine Learning"})},
                files={"file": ("lecture.pdf", b"not a pdf", "application/pdf")},
            )
        assert response.status_code == 400
        assert "magic bytes" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_upload_route_reports_duplicate_file(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_material_service] = lambda: DuplicateMaterialService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/materials/upload",
                data={"metadata": json.dumps({"owner_id": "user_demo", "collection_name": "Machine Learning"})},
                files={"file": ("lecture.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
            )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_batch_upload_route_returns_per_file_results(monkeypatch) -> None:
    monkeypatch.setenv("AGENTBOOK_TESTING", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_material_service] = lambda: FakeMaterialService()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/materials/batch_upload",
                data={
                    "metadata": json.dumps(
                        [
                            {"owner_id": "user_demo", "collection_name": "Machine Learning"},
                            {"owner_id": "user_demo", "collection_name": "Machine Learning"},
                        ]
                    )
                },
                files=[
                    ("files", ("good.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")),
                    ("files", ("bad.pdf", b"not a pdf", "application/pdf")),
                ],
            )
        assert response.status_code == 207
        body = response.json()
        assert body["data"]["results"][0]["success"] is True
        assert body["data"]["results"][1]["success"] is False
        assert "magic bytes" in body["data"]["results"][1]["error"]
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_material_status_response_schema() -> None:
    from src.schemas.material import MaterialStatusResponse

    status = MaterialStatusResponse(
        material_id="65f000000000000000000001",
        collection_id="65f000000000000000000002",
        status="indexing",
        stage="indexing",
        progress_pct=80,
    )

    assert isinstance(status, BaseModel)
    assert status.progress_pct == 80
    assert status.error_message is None


def test_upload_marks_failed_when_enqueue_fails(monkeypatch, tmp_path) -> None:
    settings = get_settings()
    settings.celery_task_always_eager = False
    settings.data_dir = tmp_path
    settings.raw_data_dir = tmp_path / "raw"
    settings.processed_data_dir = tmp_path / "processed"

    service = MaterialService(settings)
    created: dict[str, object] = {}

    async def save_collection():
        return None

    collection = SimpleNamespace(
        id=PydanticObjectId("65f000000000000000000002"),
        material_ids=[],
        updated_at=None,
        save=save_collection,
    )

    async def fake_resolve_collection(_metadata):
        return collection

    async def fake_noop(*args, **kwargs):
        return None

    def fake_move_raw_file(**kwargs):
        return "raw/user_demo/65f000000000000000000002/safe.pdf"

    async def fake_enqueue_parse_index(**kwargs):
        raise RuntimeError("broker down")

    class FakeMaterial:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.id = None
            self.created_at = utc_now()

        async def insert(self):
            self.id = PydanticObjectId("65f000000000000000000001")
            created["material"] = self

        async def save(self):
            return None

        @classmethod
        async def get(cls, _id):
            return created["material"]

    class FakePipelineJob:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.id = None

        async def insert(self):
            created["job"] = self

        async def save(self):
            return None

        @classmethod
        async def find_one(cls, *args, **kwargs):
            return created["job"]

    monkeypatch.setattr(service, "_resolve_collection", fake_resolve_collection)
    monkeypatch.setattr(service, "_ensure_not_duplicate", fake_noop)
    monkeypatch.setattr(service, "_move_raw_file", fake_move_raw_file)
    monkeypatch.setattr(service, "_enqueue_parse_index", fake_enqueue_parse_index)
    monkeypatch.setattr(material_service_module, "Material", FakeMaterial)
    monkeypatch.setattr(material_service_module, "PipelineJob", FakePipelineJob)

    metadata = {
        "owner_id": "user_demo",
        "collection_name": "Machine Learning",
        "subject": "Machine Learning",
        "language": "en",
        "modality": "mixed",
        "version": "v1.0",
    }

    from src.schemas.material import MaterialUploadMetadata

    material = asyncio.run(
        service.upload_material_from_temp(
            metadata=MaterialUploadMetadata.model_validate(metadata),
            original_filename="lecture.pdf",
            content_type="application/pdf",
            temp_path=tmp_path / "lecture.pdf",
            file_size_bytes=16,
            checksum_sha256="a" * 64,
            head=b"%PDF-1.4\n%%EOF\n",
        )
    )

    assert material.status == PipelineStatus.FAILED.value
    assert material.stage == PipelineStatus.FAILED.value


def test_upload_from_temp_saves_file_and_material_scope(monkeypatch, tmp_path) -> None:
    settings = get_settings()
    settings.data_dir = tmp_path
    settings.raw_data_dir = tmp_path / "raw"
    settings.processed_data_dir = tmp_path / "processed"

    service = MaterialService(settings)
    created: dict[str, object] = {}
    collection_id = PydanticObjectId("65f000000000000000000002")
    material_id = PydanticObjectId("65f000000000000000000001")
    payload = b"%PDF-1.4\nbody\n%%EOF\n"
    temp_path = tmp_path / "upload.tmp"
    temp_path.write_bytes(payload)

    async def save_collection():
        return None

    collection = SimpleNamespace(
        id=collection_id,
        material_ids=[],
        updated_at=None,
        save=save_collection,
    )

    async def fake_resolve_collection(_metadata):
        return collection

    async def fake_noop(*args, **kwargs):
        return None

    async def fake_enqueue_parse_index(**kwargs):
        created["enqueue"] = kwargs

    class FakeMaterial:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.id = None
            self.created_at = utc_now()

        async def insert(self):
            self.id = material_id
            created["material"] = self

        async def save(self):
            created["saved_material"] = self

    class FakePipelineJob:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        async def insert(self):
            created["job"] = self

        async def save(self):
            created["saved_job"] = self

    monkeypatch.setattr(service, "_resolve_collection", fake_resolve_collection)
    monkeypatch.setattr(service, "_ensure_not_duplicate", fake_noop)
    monkeypatch.setattr(service, "_enqueue_parse_index", fake_enqueue_parse_index)
    monkeypatch.setattr(material_service_module, "Material", FakeMaterial)
    monkeypatch.setattr(material_service_module, "PipelineJob", FakePipelineJob)

    from src.schemas.material import MaterialUploadMetadata

    response = asyncio.run(
        service.upload_material_from_temp(
            metadata=MaterialUploadMetadata.model_validate(
                {
                    "owner_id": "user_demo",
                    "collection_id": str(collection_id),
                    "subject": "Machine Learning",
                    "language": "en",
                    "modality": "mixed",
                    "version": "v1.0",
                }
            ),
            original_filename="lecture.pdf",
            content_type="Application/PDF; charset=binary",
            temp_path=temp_path,
            file_size_bytes=len(payload),
            checksum_sha256="b" * 64,
            head=payload,
        )
    )

    material = created["material"]
    job = created["job"]
    saved_path = tmp_path / response.storage_path

    assert response.status == PipelineStatus.UPLOADED.value
    assert response.stage == PipelineStatus.UPLOADED.value
    assert response.collection_id == str(collection_id)
    assert response.file_size_bytes == len(payload)
    assert temp_path.exists() is False
    assert saved_path.exists()
    assert saved_path.read_bytes() == payload
    assert saved_path.relative_to(settings.raw_data_dir)
    assert material.owner_id == "user_demo"
    assert material.collection_id == collection_id
    assert material.status == PipelineStatus.UPLOADED.value
    assert material.file_type == "pdf"
    assert material.storage_path == response.storage_path
    assert collection.material_ids == [material_id]
    assert job.status == PipelineStatus.UPLOADED.value
    assert job.stage == PipelineStatus.UPLOADED.value
