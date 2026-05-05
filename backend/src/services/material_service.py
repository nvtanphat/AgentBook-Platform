from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from beanie import PydanticObjectId
from qdrant_client import QdrantClient, models as qdrant_models

from src.core.config import Settings
from src.core.security import (
    ensure_child_path,
    safe_scope_segment,
    validate_upload_head,
    validate_upload_bytes,
)
from src.models.chunk import Chunk
from src.models.collection import KnowledgeCollection
from src.models.common import JobType, PipelineStatus, utc_now
from src.models.knowledge_graph import Entity, Event, Relation
from src.models.material import Material, MaterialPageDocument
from src.models.pipeline_job import PipelineJob
from src.models.query_log import QueryLog
from src.rag.vector_store import get_qdrant_client_for_settings
from src.schemas.material import MaterialUploadMetadata, MaterialUploadResponse

logger = logging.getLogger(__name__)

# Limits concurrent parse/index pipelines to avoid OOM from parallel model loads
# (Docling + BGE-M3 + PaddleOCR × N tasks). Shared across all MaterialService instances.
_PIPELINE_SEMAPHORE: "asyncio.Semaphore | None" = None


def _get_pipeline_semaphore() -> "asyncio.Semaphore":
    import asyncio
    global _PIPELINE_SEMAPHORE
    if _PIPELINE_SEMAPHORE is None:
        _PIPELINE_SEMAPHORE = asyncio.Semaphore(1)
    return _PIPELINE_SEMAPHORE


class MaterialService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def upload_material(
        self,
        *,
        metadata: MaterialUploadMetadata,
        original_filename: str,
        content_type: str | None,
        payload: bytes,
    ) -> MaterialUploadResponse:
        allowed_extensions = {extension.lower() for extension in self.settings.allowed_upload_extensions}
        file_type = validate_upload_bytes(
            filename=original_filename,
            content_type=content_type,
            payload=payload,
            allowed_extensions=allowed_extensions,
        )

        collection = await self._resolve_collection(metadata)
        checksum = hashlib.sha256(payload).hexdigest()
        await self._ensure_not_duplicate(owner_id=metadata.owner_id, collection_id=collection.id, checksum=checksum)
        storage_path = self._write_raw_file(
            owner_id=metadata.owner_id,
            collection_id=str(collection.id),
            checksum=checksum,
            file_type=file_type,
            payload=payload,
        )

        return await self._create_material_and_job(
            metadata=metadata,
            collection=collection,
            original_filename=original_filename,
            file_type=file_type,
            checksum=checksum,
            file_size_bytes=len(payload),
            storage_path=storage_path,
        )

    async def upload_material_from_temp(
        self,
        *,
        metadata: MaterialUploadMetadata,
        original_filename: str,
        content_type: str | None,
        temp_path: Path,
        file_size_bytes: int,
        checksum_sha256: str,
        head: bytes,
    ) -> MaterialUploadResponse:
        allowed_extensions = {extension.lower() for extension in self.settings.allowed_upload_extensions}
        file_type = validate_upload_head(
            filename=original_filename,
            content_type=content_type,
            head=head,
            allowed_extensions=allowed_extensions,
        )
        collection = await self._resolve_collection(metadata)
        await self._ensure_not_duplicate(owner_id=metadata.owner_id, collection_id=collection.id, checksum=checksum_sha256)
        storage_path = self._move_raw_file(
            owner_id=metadata.owner_id,
            collection_id=str(collection.id),
            checksum=checksum_sha256,
            file_type=file_type,
            temp_path=temp_path,
        )
        return await self._create_material_and_job(
            metadata=metadata,
            collection=collection,
            original_filename=original_filename,
            file_type=file_type,
            checksum=checksum_sha256,
            file_size_bytes=file_size_bytes,
            storage_path=storage_path,
        )

    async def _create_material_and_job(
        self,
        *,
        metadata: MaterialUploadMetadata,
        collection: KnowledgeCollection,
        original_filename: str,
        file_type: str,
        checksum: str,
        file_size_bytes: int,
        storage_path: str,
    ) -> MaterialUploadResponse:
        material = Material(
            owner_id=metadata.owner_id,
            collection_id=collection.id,
            filename=Path(storage_path).name,
            original_name=original_filename,
            file_type=file_type,
            modality=metadata.modality,
            language=metadata.language,
            subject=metadata.subject,
            topic=metadata.topic,
            version=metadata.version,
            checksum_sha256=checksum,
            file_size_bytes=file_size_bytes,
            storage_path=storage_path,
            status=PipelineStatus.UPLOADED.value,
            parse_version=self.settings.parse_version,
            chunk_version=self.settings.chunk_version,
            embedding_version=self.settings.embedding_version,
            index_version=self.settings.index_version,
            extra_metadata=self._build_extra_metadata(metadata),
        )
        await material.insert()

        collection.material_ids.append(material.id)
        collection.updated_at = material.created_at
        await collection.save()

        job = PipelineJob(
            material_id=material.id,
            job_id=str(uuid4()),
            job_type=JobType.PARSE_INDEX.value,
            status=PipelineStatus.UPLOADED.value,
            stage=PipelineStatus.UPLOADED.value,
        )
        await job.insert()
        try:
            await self._enqueue_parse_index(material_id=str(material.id), job_id=job.job_id)
        except Exception:
            logger.exception(
                "Failed to enqueue parse/index task",
                extra={"material_id": str(material.id), "job_id": job.job_id},
            )
            await self._mark_enqueue_failed(material=material, job=job, error="Failed to enqueue parse/index task")

        return MaterialUploadResponse(
            material_id=str(material.id),
            doc_id=str(material.id),
            collection_id=str(collection.id),
            job_id=job.job_id,
            status=material.status,
            stage=job.stage,
            filename=material.filename,
            original_name=material.original_name,
            checksum_sha256=material.checksum_sha256,
            file_size_bytes=material.file_size_bytes,
            storage_path=material.storage_path,
        )

    async def _ensure_not_duplicate(self, *, owner_id: str, collection_id: PydanticObjectId, checksum: str) -> None:
        existing = await Material.find_one(
            Material.owner_id == owner_id,
            Material.collection_id == collection_id,
            Material.checksum_sha256 == checksum,
        )
        if existing is not None:
            raise ValueError("File already exists in this collection")

    async def _resolve_collection(self, metadata: MaterialUploadMetadata) -> KnowledgeCollection:
        if metadata.collection_id:
            try:
                collection_id = PydanticObjectId(metadata.collection_id)
            except Exception as exc:
                raise ValueError("collection_id must be a valid ObjectId") from exc
            collection = await KnowledgeCollection.get(collection_id)
            if collection is None or collection.owner_id != metadata.owner_id:
                raise LookupError("Collection was not found for this owner")
            return collection

        collection = KnowledgeCollection(
            name=metadata.collection_name or metadata.subject or "Default Collection",
            subject=metadata.subject,
            description=metadata.collection_description,
            owner_id=metadata.owner_id,
        )
        await collection.insert()
        return collection

    def _write_raw_file(
        self,
        *,
        owner_id: str,
        collection_id: str,
        checksum: str,
        file_type: str,
        payload: bytes,
    ) -> str:
        owner_dir = safe_scope_segment(owner_id)
        collection_dir = safe_scope_segment(collection_id)
        filename = f"{checksum[:16]}-{uuid4().hex}.{file_type}"
        root = self.settings.raw_data_dir
        target_dir = root / owner_dir / collection_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = ensure_child_path(root, target_dir / filename)
        target_path.write_bytes(payload)
        return str(target_path.relative_to(root.parent)).replace("\\", "/")

    def _move_raw_file(
        self,
        *,
        owner_id: str,
        collection_id: str,
        checksum: str,
        file_type: str,
        temp_path: Path,
    ) -> str:
        owner_dir = safe_scope_segment(owner_id)
        collection_dir = safe_scope_segment(collection_id)
        filename = f"{checksum[:16]}-{uuid4().hex}.{file_type}"
        root = self.settings.raw_data_dir
        target_dir = root / owner_dir / collection_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = ensure_child_path(root, target_dir / filename)
        temp_path.replace(target_path)
        return str(target_path.relative_to(root.parent)).replace("\\", "/")

    @staticmethod
    async def _mark_enqueue_failed(*, material: Material, job: PipelineJob, error: str) -> None:
        material.status = PipelineStatus.FAILED.value
        material.failed_stage = "enqueue"
        material.error_message = error
        material.updated_at = utc_now()
        job.status = PipelineStatus.FAILED.value
        job.stage = PipelineStatus.FAILED.value
        job.failed_stage = "enqueue"
        job.last_error = error
        job.finished_at = utc_now()
        await material.save()
        await job.save()

    async def _enqueue_parse_index(self, *, material_id: str, job_id: str) -> None:
        if self.settings.celery_task_always_eager:
            import asyncio
            from src.services.parse_index_pipeline import ParseIndexPipeline

            async def _run_bg() -> None:
                async with _get_pipeline_semaphore():
                    try:
                        await ParseIndexPipeline(settings=self.settings).run(
                            material_id=material_id, job_id=job_id
                        )
                    except Exception:
                        logger.exception(
                            "Parse/index background task failed",
                            extra={"material_id": material_id, "job_id": job_id},
                        )

            asyncio.create_task(_run_bg())
            return

        try:
            from src.tasks.celery_tasks import parse_and_index_material_task

            parse_and_index_material_task.delay(material_id, job_id)
        except Exception as exc:
            logger.warning(
                "Failed to enqueue parse/index task — job stays at UPLOADED and must be retried manually. "
                "Set AGENTBOOK_CELERY_TASK_ALWAYS_EAGER=true to run synchronously without a broker.",
                extra={"material_id": material_id, "job_id": job_id, "error": str(exc)},
            )

    async def retry_material(self, *, material_id: str, owner_id: str) -> dict:
        try:
            mid = PydanticObjectId(material_id)
        except Exception as exc:
            raise ValueError("material_id must be a valid ObjectId") from exc

        material = await Material.get(mid)
        if material is None or material.owner_id != owner_id:
            raise LookupError("Material not found for this owner")

        non_retryable = {PipelineStatus.INDEXED.value}
        if material.status in non_retryable:
            raise ValueError(f"Cannot retry material in status '{material.status}'")

        material.status = PipelineStatus.UPLOADED.value
        material.failed_stage = None
        material.error_message = None
        material.updated_at = utc_now()
        await material.save()

        job = PipelineJob(
            material_id=material.id,
            job_id=str(uuid4()),
            job_type=JobType.PARSE_INDEX.value,
            status=PipelineStatus.UPLOADED.value,
            stage=PipelineStatus.UPLOADED.value,
        )
        await job.insert()
        await self._enqueue_parse_index(material_id=str(material.id), job_id=job.job_id)
        return {"material_id": material_id, "job_id": job.job_id, "status": material.status}

    async def delete_material(self, *, material_id: str, owner_id: str) -> dict[str, int]:
        """Delete a single material and all its associated chunks, vectors, and jobs."""
        try:
            mid = PydanticObjectId(material_id)
        except Exception as exc:
            raise ValueError("material_id must be a valid ObjectId") from exc

        material = await Material.get(mid)
        if material is None or material.owner_id != owner_id:
            raise LookupError("Material not found for this owner")

        # Remove Qdrant vectors for this material
        qdrant: QdrantClient = get_qdrant_client_for_settings(self.settings)
        try:
            qdrant.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="material_id",
                                match=qdrant_models.MatchValue(value=material_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant delete failed for material", extra={"material_id": material_id, "error": str(exc)})

        # Delete raw file from disk
        if material.storage_path:
            raw_file = self.settings.raw_data_dir.parent / material.storage_path
            try:
                raw_file.unlink(missing_ok=True)
            except Exception:
                pass

        # Delete parsed JSON artifact (processed data) from disk
        self._delete_data_file(material.extra_metadata.get("parsed_artifact_path"))

        # Remove material_id from collection's material_ids list
        collection = await KnowledgeCollection.get(material.collection_id)
        if collection and mid in collection.material_ids:
            collection.material_ids.remove(mid)
            await collection.save()

        # Delete MongoDB documents (chunks, jobs, graph nodes, material)
        r_chunks = await Chunk.find(Chunk.material_id == mid).delete()
        r_pages = await MaterialPageDocument.find(MaterialPageDocument.material_id == mid).delete()
        r_jobs = await PipelineJob.find(PipelineJob.material_id == mid).delete()
        # Cascade-delete graph nodes referencing this material to prevent orphan refs
        r_entities = await Entity.find({"mention_refs.material_id": mid}).delete()
        r_events = await Event.find({"evidence_refs.material_id": mid}).delete()
        r_relations = await Relation.find({"evidence_refs.material_id": mid}).delete()
        await material.delete()

        counts = {
            "chunks": getattr(r_chunks, "deleted_count", 0),
            "pages": getattr(r_pages, "deleted_count", 0),
            "jobs": getattr(r_jobs, "deleted_count", 0),
            "entities": getattr(r_entities, "deleted_count", 0),
            "events": getattr(r_events, "deleted_count", 0),
            "relations": getattr(r_relations, "deleted_count", 0),
            "materials": 1,
        }
        logger.info("Material deleted", extra={"material_id": material_id, "owner_id": owner_id, **counts})
        return counts

    async def delete_collection(self, *, collection_id: str, owner_id: str) -> dict[str, int]:
        """Delete a collection and all its associated data (materials, chunks, vectors, graph nodes)."""
        try:
            cid = PydanticObjectId(collection_id)
        except Exception as exc:
            raise ValueError("collection_id must be a valid ObjectId") from exc

        collection = await KnowledgeCollection.get(cid)
        if collection is None or collection.owner_id != owner_id:
            raise LookupError("Collection not found for this owner")

        # Remove Qdrant vectors for this collection
        qdrant: QdrantClient = get_qdrant_client_for_settings(self.settings)
        try:
            qdrant.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="collection_id",
                                match=qdrant_models.MatchValue(value=collection_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant delete failed for collection", extra={"collection_id": collection_id, "error": str(exc)})

        # Delete raw files and parsed JSON artifacts from disk
        materials = await Material.find(Material.collection_id == cid, Material.owner_id == owner_id).to_list()
        for material in materials:
            if material.storage_path:
                raw_file = self.settings.raw_data_dir.parent / material.storage_path
                try:
                    raw_file.unlink(missing_ok=True)
                except Exception:
                    pass
            self._delete_data_file(material.extra_metadata.get("parsed_artifact_path"))

        # Delete MongoDB documents
        from beanie.operators import In as BIn

        r_chunks = await Chunk.find(Chunk.collection_id == cid).delete()
        r_pages = await MaterialPageDocument.find(MaterialPageDocument.collection_id == cid).delete()
        r_materials = await Material.find(Material.collection_id == cid).delete()
        if materials:
            material_ids = [m.id for m in materials]
            r_jobs = await PipelineJob.find(BIn(PipelineJob.material_id, material_ids)).delete()
        else:
            r_jobs = None
        r_entities = await Entity.find(Entity.collection_id == cid).delete()
        r_events = await Event.find(Event.collection_id == cid).delete()
        r_relations = await Relation.find(Relation.collection_id == cid).delete()
        r_logs = await QueryLog.find(QueryLog.collection_id == cid).delete()
        await collection.delete()

        counts = {
            "chunks": getattr(r_chunks, "deleted_count", 0),
            "pages": getattr(r_pages, "deleted_count", 0),
            "materials": getattr(r_materials, "deleted_count", 0),
            "jobs": getattr(r_jobs, "deleted_count", 0),
            "entities": getattr(r_entities, "deleted_count", 0),
            "events": getattr(r_events, "deleted_count", 0),
            "relations": getattr(r_relations, "deleted_count", 0),
            "query_logs": getattr(r_logs, "deleted_count", 0),
        }
        logger.info("Collection deleted", extra={"collection_id": collection_id, "owner_id": owner_id, **counts})
        return counts

    @staticmethod
    def _build_extra_metadata(metadata: MaterialUploadMetadata) -> dict[str, object]:
        extra = dict(metadata.extra_metadata)
        if metadata.source_type:
            extra["source_type"] = metadata.source_type
        return extra

    def _delete_data_file(self, stored_path: object) -> None:
        if not isinstance(stored_path, str) or not stored_path.strip():
            return
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = self.settings.data_dir / candidate
            if not candidate.exists():
                candidate = self.settings.data_dir.parent / stored_path
        try:
            target = candidate.resolve()
            data_root = self.settings.data_dir.resolve()
            target.relative_to(data_root)
        except Exception:
            logger.warning("Refusing to delete artifact outside data dir", extra={"path": stored_path})
            return
        try:
            target.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to delete artifact", extra={"path": str(target), "error": str(exc)})
