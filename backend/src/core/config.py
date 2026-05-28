from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_dotenv_into_environ() -> None:
    env_path = project_root() / "backend" / ".env"
    if not env_path.exists():
        env_path = project_root() / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val


_load_dotenv_into_environ()


def load_yaml_config(name: str) -> dict[str, Any]:
    config_path = project_root() / "config" / name
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")
    return data


def env_value(name: str, fallback: Any) -> Any:
    return os.getenv(f"AGENTBOOK_{name}", fallback)


def env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(f"AGENTBOOK_{name}")
    if raw is None:
        return fallback
    return raw.strip().lower() not in ("false", "0", "no", "")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENTBOOK_",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    app_name: str = "Noelys"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    testing: bool = False
    api_auth_enabled: bool = False
    api_key: str | None = None

    mongodb_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mongodb_uri", "MONGODB_URI", "AGENTBOOK_MONGODB_URI"),
    )
    mongodb_database: str = "agentbook"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "agentbook_chunks"
    qdrant_timeout_seconds: int = 60

    redis_url: str = "redis://localhost:6379/0"

    data_dir: Path = Field(default_factory=lambda: project_root() / "data")
    raw_data_dir: Path = Field(default_factory=lambda: project_root() / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: project_root() / "data" / "processed")

    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "pptx", "png", "jpg", "jpeg", "csv", "xlsx", "xls"]
    )
    max_upload_size_mb: int = 20
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ])

    parse_version: str = "docling-2026-04"
    chunk_version: str = "semantic_v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dense_size: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024
    embedding_use_fp16: bool = False
    normalize_embeddings: bool = True
    embedding_version: str = "bge-m3-v1"
    index_version: str = "qdrant_dense_sparse_v1"
    index_batch_size: int = 64
    llm_default_provider: str = "local"
    llm_local_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1024
    llm_timeout_seconds: float = 180.0
    reranker_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_max_pairs: int = 100
    dense_top_k: int = 20
    sparse_top_k: int = 20
    graph_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    rerank_input_k: int = 20
    max_chunks_per_doc: int = 3
    graph_max_hops: int = 2
    agentic_rag_enabled: bool = False
    agentic_planner_llm_enabled: bool = False
    agentic_max_retrieval_iterations: int = 2
    agentic_anaphora_resolution_enabled: bool = True
    # Coordinator-only — confidence threshold below which the legacy critic
    # agent fires (after guardrails). Set in retrieval_config.yaml under
    # `retrieval.agentic_critic_activation_confidence`.
    agentic_critic_activation_confidence: float = 0.65
    # When false, skip the post-synthesis CriticAgent loop entirely. Saves
    # 30-90s per query at the cost of one self-review pass.
    agentic_critic_enabled: bool = True
    multi_query_enabled: bool = False
    smart_reranker_enabled: bool = False
    smart_reranker_threshold: float = 0.7
    hyde_enabled: bool = False
    llm_router_enabled: bool = False
    crag_evaluator_enabled: bool = False
    crag_correct_threshold: float = 0.55
    crag_incorrect_threshold: float = 0.25
    self_rag_reflection_enabled: bool = False
    min_reranker_score: float = 0.35
    min_evidence_confidence: float = 0.55
    min_graph_confidence: float = 0.55
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_always_eager: bool = False
    chunk_strategy: str = "semantic"
    contextual_retrieval_enabled: bool = True
    contextual_retrieval_concurrency: int = 4
    chunk_target_token_count: int = 512
    chunk_min_token_count: int = 100
    chunk_overlap_token_count: int = 50
    chunk_max_blocks_per_chunk: int = 8
    semantic_chunk_breakpoint_percentile: float = 95.0
    min_ocr_text_quality: float = 0.35
    warn_ocr_text_quality: float = 0.55
    min_handwriting_quality_score: float = 0.72
    min_handwriting_confidence: float = 0.8
    min_blur_variance: float = 80.0
    min_brightness: float = 45.0
    max_brightness: float = 230.0
    min_contrast: float = 18.0
    max_abs_skew_degrees: float = 12.0

    # Structure-adaptive visualization config (see config/viz_config.yaml).
    # Loaded as a nested dict (ladders/patterns/thresholds) — consumed by the
    # structure detector. Empty dict = detector uses built-in defaults.
    viz_config: dict = Field(default_factory=dict)

    # Entity extraction config (see config/extraction_config.yaml)
    extraction_mode: str = "dynamic"  # simple | dynamic | strict
    extraction_default_entity_types: list[str] = Field(
        default_factory=lambda: [
            "concept", "person", "organization", "location",
            "event", "artifact", "time", "quantity",
        ]
    )
    extraction_output_format: str = "json"  # json | delimiter
    extraction_max_gleanings: int = 0
    extraction_domain_hint_field: str = "subject"
    extraction_few_shots: list[dict] = Field(default_factory=list)

    # Knowledge-graph endpoint thresholds + fetch caps (see retrieval_config.yaml → graph)
    graph_fuzzy_dedup_threshold: float = 0.88
    graph_semantic_dedup_threshold: float = 0.82
    graph_display_min_confidence: float = 0.5
    graph_display_min_mentions: int = 2
    graph_max_entities_fetch: int = 400
    graph_max_relations_fetch: int = 400
    graph_max_visible_nodes: int = 150
    graph_focus_primary_cap: int = 15
    graph_focus_neighbor_cap: int = 12
    graph_focus_fallback_cap: int = 20
    graph_mindmap_entity_fetch: int = 160
    graph_mindmap_entity_cap: int = 60
    graph_semantic_relation_max_concepts: int = 25
    graph_semantic_relation_max_passages: int = 18
    graph_semantic_relation_max_passage_chars: int = 600

    # Phase B — Adaptive Retrieval Budget (skip sparse + graph on easy factuals)
    adaptive_retrieval_enabled: bool = True
    adaptive_dense_skip_threshold: float = 0.55
    adaptive_strong_hits_required: int = 3
    adaptive_strong_hit_min_score: float = 0.35
    adaptive_eligible_routes: list[str] = Field(
        default_factory=lambda: ["factual", "general", "claim_check"]
    )

    # Sentence-level Evidence Coverage (SLEC) — Adaptive Evidence-Guided RAG core
    slec_enabled: bool = True
    slec_supported_threshold: float = 0.55
    slec_partial_threshold: float = 0.30
    slec_refuse_below: float = 0.40
    slec_drop_unsupported: bool = True
    slec_min_sentence_chars: int = 12
    slec_max_sentences: int = 24
    slec_skip_routes: list[str] = Field(default_factory=lambda: ["claim_check"])

    # Visual embedding (SigLIP)
    visual_embedding_enabled: bool = False
    visual_embedding_model: str = "google/siglip-base-patch16-224"
    visual_embedding_device: str = "cpu"
    visual_embedding_dense_size: int = 768
    visual_embedding_batch_size: int = 4
    visual_embedding_backend: str = "pytorch"
    qdrant_visual_collection_name: str = "agentbook_visual"
    visual_retrieval_top_k: int = 3
    lexical_fallback_multiplier: int = 3
    chunk_min_content_chars: int = 50
    figure_captioner_max_workers: int = 4
    ollama_caption_timeout_seconds: float = 300.0
    figure_image_max_side_px: int = 1024
    memory_max_turns: int = 10

    ocr_text_detection_model_name: str = "PP-OCRv5_mobile_det"
    ocr_text_recognition_model_name: str | None = None
    ocr_text_det_limit_side_len: int = 1280
    ocr_text_det_limit_type: str = "max"
    ocr_rec_score_threshold: float = 0.5
    ocr_enable_grayscale_variant: str = "auto"
    ocr_grayscale_trigger_confidence: float = 0.85
    ocr_recognition_engine: str = "easyocr"  # easyocr | vietocr (vi-only)
    ocr_vietocr_model_name: str = "vgg_transformer"
    ocr_vietocr_device: str = "cpu"
    pdf_render_scale: float = 1.5

    # Audio (faster-whisper)
    audio_whisper_model: str = "small"
    audio_whisper_device: str = "cpu"
    audio_whisper_compute_type: str = "int8"
    audio_whisper_beam_size: int = 5
    audio_whisper_vad_filter: bool = True
    audio_chunk_target_seconds: float = 45.0
    audio_chunk_min_seconds: float = 10.0

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @model_validator(mode="after")
    def validate_chunking_config(self) -> "Settings":
        """Validate chunking configuration constraints."""
        if self.app_env.lower() == "production":
            if not self.api_auth_enabled:
                raise ValueError("api_auth_enabled must be true when app_env is production")
            if not self.api_key:
                raise ValueError("api_key must be configured when app_env is production")
        for name, value, upper in [
            ("dense_top_k", self.dense_top_k, 100),
            ("sparse_top_k", self.sparse_top_k, 100),
            ("graph_top_k", self.graph_top_k, 50),
            ("final_top_k", self.final_top_k, 20),
            ("rerank_input_k", self.rerank_input_k, 100),
            ("reranker_max_pairs", self.reranker_max_pairs, 200),
        ]:
            if not 1 <= int(value) <= upper:
                raise ValueError(f"{name} ({value}) must be between 1 and {upper}")
        if self.llm_max_output_tokens < 1 or self.llm_max_output_tokens > 8192:
            raise ValueError("llm_max_output_tokens must be between 1 and 8192")
        if self.chunk_min_token_count >= self.chunk_target_token_count:
            raise ValueError(
                f"chunk_min_token_count ({self.chunk_min_token_count}) must be less than "
                f"chunk_target_token_count ({self.chunk_target_token_count})"
            )
        if self.chunk_target_token_count > self.embedding_max_length:
            raise ValueError(
                f"chunk_target_token_count ({self.chunk_target_token_count}) must not exceed "
                f"embedding_max_length ({self.embedding_max_length})"
            )
        if self.chunk_overlap_token_count >= self.chunk_min_token_count:
            raise ValueError(
                f"chunk_overlap_token_count ({self.chunk_overlap_token_count}) must be less than "
                f"chunk_min_token_count ({self.chunk_min_token_count})"
            )
        if self.chunk_overlap_token_count < 0:
            raise ValueError(f"chunk_overlap_token_count ({self.chunk_overlap_token_count}) must be non-negative")
        if self.chunk_min_token_count < 1:
            raise ValueError(f"chunk_min_token_count ({self.chunk_min_token_count}) must be at least 1")
        if self.chunk_max_blocks_per_chunk < 1:
            raise ValueError(f"chunk_max_blocks_per_chunk ({self.chunk_max_blocks_per_chunk}) must be at least 1")
        if not 0 < self.semantic_chunk_breakpoint_percentile <= 100:
            raise ValueError(
                f"semantic_chunk_breakpoint_percentile ({self.semantic_chunk_breakpoint_percentile}) "
                f"must be between 0 and 100"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_config = load_yaml_config("model_config.yaml")
    retrieval_config = load_yaml_config("retrieval_config.yaml")
    guardrails_config = load_yaml_config("guardrails_config.yaml")

    upload_config = guardrails_config.get("upload", {})
    refusal_config = guardrails_config.get("refusal", {})
    image_quality_config = guardrails_config.get("image_quality", {})
    slec_config = guardrails_config.get("sentence_coverage", {})
    embedding_config = model_config.get("embedding", {})
    llm_config = model_config.get("llm", {})
    reranker_config = model_config.get("reranker", {})
    versions = model_config.get("versions", {})
    ocr_config = model_config.get("ocr", {})
    pdf_config = model_config.get("pdf", {})
    figure_captioner_config = model_config.get("figure_captioner", {})
    memory_config = model_config.get("memory", {})
    visual_config = model_config.get("visual_embedding", {})
    audio_config = model_config.get("audio", {})
    qdrant_config = retrieval_config.get("qdrant", {})
    retrieval_section = retrieval_config.get("retrieval", {})
    chunking_config = retrieval_config.get("chunking", {})
    adaptive_retrieval_config = retrieval_config.get("adaptive_retrieval", {})
    crag_config = retrieval_config.get("crag", {})
    graph_config = retrieval_config.get("graph", {})
    graph_semrel_config = graph_config.get("semantic_relation", {})
    try:
        extraction_config = load_yaml_config("extraction_config.yaml").get("extraction", {})
    except FileNotFoundError:
        extraction_config = {}
    try:
        viz_config = load_yaml_config("viz_config.yaml").get("visualization", {})
    except FileNotFoundError:
        viz_config = {}

    return Settings(
        max_upload_size_mb=upload_config.get("max_file_size_mb", 20),
        allowed_upload_extensions=upload_config.get(
            "allowed_extensions", ["pdf", "docx", "pptx", "png", "jpg", "jpeg", "csv", "xlsx", "xls"]
        ),
        embedding_model=embedding_config.get("model_name", "BAAI/bge-m3"),
        embedding_dense_size=embedding_config.get("dense_size", 1024),
        embedding_device=env_value("EMBEDDING_DEVICE", embedding_config.get("device", "cpu")),
        embedding_batch_size=int(env_value("EMBEDDING_BATCH_SIZE", embedding_config.get("batch_size", 8))),
        embedding_max_length=embedding_config.get("max_length", 1024),
        embedding_use_fp16=env_bool("EMBEDDING_USE_FP16", embedding_config.get("use_fp16", False)),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        embedding_version=embedding_config.get("embedding_version", "bge-m3-v1"),
        qdrant_collection_name=qdrant_config.get("collection_name", "agentbook_chunks"),
        parse_version=versions.get("parse_version", "docling-2026-04"),
        chunk_version=versions.get("chunk_version", "semantic_v1"),
        index_version=versions.get("index_version", "qdrant_dense_sparse_v1"),
        index_batch_size=qdrant_config.get("index_batch_size", 64),
        llm_default_provider=env_value("LLM_DEFAULT_PROVIDER", llm_config.get("default_provider", "local")),
        llm_local_model=env_value("LLM_LOCAL_MODEL", llm_config.get("local_model", "qwen2.5:3b")),
        ollama_base_url=env_value("OLLAMA_BASE_URL", llm_config.get("ollama_base_url", "http://localhost:11434")),
        openai_base_url=env_value("OPENAI_BASE_URL", llm_config.get("openai_base_url", "https://api.openai.com/v1")),
        openai_model=env_value("OPENAI_MODEL", llm_config.get("openai_model", "gpt-4o-mini")),
        llm_temperature=llm_config.get("temperature", 0.1),
        llm_max_output_tokens=llm_config.get("max_output_tokens", 1024),
        llm_timeout_seconds=float(env_value("LLM_TIMEOUT_SECONDS", llm_config.get("timeout_seconds", 180.0))),
        reranker_enabled=env_bool("RERANKER_ENABLED", reranker_config.get("enabled", True)),
        reranker_model_name=reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3"),
        reranker_device=reranker_config.get("device", "cpu"),
        reranker_max_pairs=int(reranker_config.get("max_pairs", 80)),
        dense_top_k=retrieval_section.get("dense_top_k", 20),
        sparse_top_k=retrieval_section.get("sparse_top_k", 20),
        graph_top_k=retrieval_section.get("graph_top_k", 10),
        final_top_k=retrieval_section.get("final_top_k", retrieval_section.get("top_k", 5)),
        rrf_k=retrieval_section.get("rrf_k", 60),
        rerank_input_k=retrieval_section.get("rerank_input_k", 15),
        graph_max_hops=min(int(retrieval_section.get("graph_max_hops", 2)), 2),
        agentic_rag_enabled=env_bool("AGENTIC_RAG_ENABLED", retrieval_section.get("agentic_rag_enabled", False)),
        agentic_planner_llm_enabled=env_bool("AGENTIC_PLANNER_LLM_ENABLED", retrieval_section.get("agentic_planner_llm_enabled", False)),
        agentic_max_retrieval_iterations=int(retrieval_section.get("agentic_max_retrieval_iterations", 2)),
        agentic_anaphora_resolution_enabled=bool(retrieval_section.get("agentic_anaphora_resolution_enabled", True)),
        agentic_critic_activation_confidence=float(retrieval_section.get("agentic_critic_activation_confidence", 0.65)),
        agentic_critic_enabled=env_bool("AGENTIC_CRITIC_ENABLED", retrieval_section.get("agentic_critic_enabled", True)),
        crag_evaluator_enabled=env_bool("CRAG_EVALUATOR_ENABLED", crag_config.get("evaluator_enabled", False)),
        crag_correct_threshold=float(crag_config.get("correct_threshold", 0.55)),
        crag_incorrect_threshold=float(crag_config.get("incorrect_threshold", 0.25)),
        multi_query_enabled=env_bool("MULTI_QUERY_ENABLED", retrieval_section.get("multi_query_enabled", False)),
        api_auth_enabled=env_bool("API_AUTH_ENABLED", str(env_value("APP_ENV", "development")).lower() == "production"),
        api_key=env_value("API_KEY", None),
        min_ocr_text_quality=refusal_config.get("min_ocr_text_quality", 0.35),
        warn_ocr_text_quality=refusal_config.get("warn_ocr_text_quality", 0.55),
        min_reranker_score=refusal_config.get("min_reranker_score", 0.35),
        min_evidence_confidence=refusal_config.get("min_evidence_confidence", 0.55),
        min_graph_confidence=refusal_config.get("min_graph_confidence", 0.55),
        chunk_strategy=chunking_config.get("strategy", "semantic"),
        contextual_retrieval_enabled=env_bool("CONTEXTUAL_RETRIEVAL_ENABLED", chunking_config.get("contextual_retrieval_enabled", True)),
        contextual_retrieval_concurrency=int(chunking_config.get("contextual_retrieval_concurrency", 4)),
        chunk_target_token_count=chunking_config.get("target_token_count", 512),
        chunk_min_token_count=chunking_config.get("min_token_count", 100),
        chunk_overlap_token_count=chunking_config.get("overlap_token_count", 50),
        chunk_max_blocks_per_chunk=chunking_config.get("max_blocks_per_chunk", 8),
        semantic_chunk_breakpoint_percentile=float(chunking_config.get("breakpoint_percentile", 95.0)),
        adaptive_retrieval_enabled=env_bool(
            "ADAPTIVE_RETRIEVAL_ENABLED",
            adaptive_retrieval_config.get("enabled", True),
        ),
        adaptive_dense_skip_threshold=float(
            adaptive_retrieval_config.get("dense_skip_threshold", 0.55)
        ),
        adaptive_strong_hits_required=int(
            adaptive_retrieval_config.get("strong_hits_required", 3)
        ),
        adaptive_strong_hit_min_score=float(
            adaptive_retrieval_config.get("strong_hit_min_score", 0.35)
        ),
        adaptive_eligible_routes=list(
            adaptive_retrieval_config.get(
                "eligible_routes", ["factual", "general", "claim_check"]
            )
        ),
        slec_enabled=env_bool("SLEC_ENABLED", slec_config.get("enabled", True)),
        slec_supported_threshold=float(slec_config.get("supported_threshold", 0.55)),
        slec_partial_threshold=float(slec_config.get("partial_threshold", 0.30)),
        slec_refuse_below=float(slec_config.get("refuse_below", 0.40)),
        slec_drop_unsupported=bool(slec_config.get("drop_unsupported", True)),
        slec_min_sentence_chars=int(slec_config.get("min_sentence_chars", 12)),
        slec_max_sentences=int(slec_config.get("max_sentences", 24)),
        slec_skip_routes=list(slec_config.get("skip_routes", ["claim_check"])),
        min_handwriting_quality_score=refusal_config.get("min_handwriting_quality_score", 0.72),
        min_handwriting_confidence=refusal_config.get("min_handwriting_confidence", 0.8),
        min_blur_variance=image_quality_config.get("min_blur_variance", 80.0),
        min_brightness=image_quality_config.get("min_brightness", 45.0),
        max_brightness=image_quality_config.get("max_brightness", 230.0),
        min_contrast=image_quality_config.get("min_contrast", 18.0),
        max_abs_skew_degrees=image_quality_config.get("max_abs_skew_degrees", 12.0),
        audio_whisper_model=audio_config.get("whisper_model", "small"),
        audio_whisper_device=audio_config.get("whisper_device", "cpu"),
        audio_whisper_compute_type=audio_config.get("whisper_compute_type", "int8"),
        audio_whisper_beam_size=int(audio_config.get("whisper_beam_size", 5)),
        audio_whisper_vad_filter=bool(audio_config.get("whisper_vad_filter", True)),
        audio_chunk_target_seconds=float(audio_config.get("audio_chunk_target_seconds", 45.0)),
        audio_chunk_min_seconds=float(audio_config.get("audio_chunk_min_seconds", 10.0)),
        ocr_text_detection_model_name=ocr_config.get("text_detection_model_name", "PP-OCRv5_mobile_det"),
        ocr_text_recognition_model_name=ocr_config.get("text_recognition_model_name"),
        ocr_text_det_limit_side_len=int(ocr_config.get("text_det_limit_side_len", 1280)),
        ocr_text_det_limit_type=ocr_config.get("text_det_limit_type", "max"),
        ocr_rec_score_threshold=float(ocr_config.get("rec_score_threshold", 0.5)),
        ocr_enable_grayscale_variant=str(ocr_config.get("enable_grayscale_variant", "auto")).lower(),
        ocr_grayscale_trigger_confidence=float(ocr_config.get("grayscale_trigger_confidence", 0.85)),
        ocr_recognition_engine=str(ocr_config.get("recognition_engine", "easyocr")).lower(),
        ocr_vietocr_model_name=str(ocr_config.get("vietocr_model_name", "vgg_transformer")),
        ocr_vietocr_device=env_value("OCR_VIETOCR_DEVICE", ocr_config.get("vietocr_device", "cpu")),
        pdf_render_scale=float(pdf_config.get("render_scale", 1.5)),
        visual_embedding_enabled=env_bool(
            "VISUAL_EMBEDDING_ENABLED", visual_config.get("enabled", False)
        ),
        visual_embedding_model=visual_config.get(
            "model", "google/siglip-base-patch16-224"
        ),
        visual_embedding_device=env_value(
            "VISUAL_EMBEDDING_DEVICE", visual_config.get("device", "cpu")
        ),
        visual_embedding_dense_size=int(visual_config.get("dense_size", 768)),
        visual_embedding_batch_size=int(
            env_value(
                "VISUAL_EMBEDDING_BATCH_SIZE", visual_config.get("batch_size", 4)
            )
        ),
        visual_embedding_backend=visual_config.get("embedding_backend", "pytorch"),
        qdrant_visual_collection_name=qdrant_config.get(
            "visual_collection_name", "agentbook_visual"
        ),
        visual_retrieval_top_k=int(retrieval_config.get("visual_retrieval_top_k", 3)),
        lexical_fallback_multiplier=int(retrieval_section.get("lexical_fallback_multiplier", 3)),
        chunk_min_content_chars=int(chunking_config.get("min_content_chars", 50)),
        figure_captioner_max_workers=int(figure_captioner_config.get("max_workers", 4)),
        ollama_caption_timeout_seconds=float(figure_captioner_config.get("ollama_timeout_seconds", 300.0)),
        figure_image_max_side_px=int(figure_captioner_config.get("image_max_side_px", 1024)),
        memory_max_turns=int(memory_config.get("max_turns", 10)),
        self_rag_reflection_enabled=env_bool(
            "SELF_RAG_REFLECTION_ENABLED", llm_config.get("self_rag_reflection_enabled", False)
        ),
        graph_fuzzy_dedup_threshold=float(graph_config.get("fuzzy_dedup_threshold", 0.88)),
        graph_semantic_dedup_threshold=float(graph_config.get("semantic_dedup_threshold", 0.82)),
        graph_display_min_confidence=float(graph_config.get("display_min_confidence", 0.5)),
        graph_display_min_mentions=int(graph_config.get("display_min_mentions", 2)),
        graph_max_entities_fetch=int(graph_config.get("max_entities_fetch", 400)),
        graph_max_relations_fetch=int(graph_config.get("max_relations_fetch", 400)),
        graph_max_visible_nodes=int(graph_config.get("max_visible_nodes", 150)),
        graph_focus_primary_cap=int(graph_config.get("focus_primary_cap", 15)),
        graph_focus_neighbor_cap=int(graph_config.get("focus_neighbor_cap", 12)),
        graph_focus_fallback_cap=int(graph_config.get("focus_fallback_cap", 20)),
        graph_mindmap_entity_fetch=int(graph_config.get("mindmap_entity_fetch", 160)),
        graph_mindmap_entity_cap=int(graph_config.get("mindmap_entity_cap", 60)),
        graph_semantic_relation_max_concepts=int(graph_semrel_config.get("max_concepts", 25)),
        graph_semantic_relation_max_passages=int(graph_semrel_config.get("max_passages", 18)),
        graph_semantic_relation_max_passage_chars=int(graph_semrel_config.get("max_passage_chars", 600)),
        extraction_mode=str(extraction_config.get("mode", "dynamic")).lower(),
        extraction_default_entity_types=list(
            extraction_config.get(
                "default_entity_types",
                ["concept", "person", "organization", "location",
                 "event", "artifact", "time", "quantity"],
            )
        ),
        extraction_output_format=str(extraction_config.get("output_format", "json")).lower(),
        extraction_max_gleanings=int(extraction_config.get("max_gleanings", 0)),
        extraction_domain_hint_field=str(extraction_config.get("domain_hint_field", "subject")),
        extraction_few_shots=list(extraction_config.get("few_shots", [])),
        viz_config=dict(viz_config),
    )
