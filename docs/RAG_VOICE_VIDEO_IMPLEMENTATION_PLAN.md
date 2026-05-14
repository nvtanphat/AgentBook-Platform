# Ke hoach trien khai RAG Voice va RAG Video cho Noelys

Ngay lap: 2026-05-10

Tai lieu nay de xuat cach mo rong Noelys tu RAG tai lieu sang RAG cho voice/audio va video. Ke hoach duoc viet theo hien trang codebase hien tai: FastAPI backend, MongoDB/Beanie, Qdrant, BGE-M3 embeddings, reranker, Graph RAG, Evidence Workspace va frontend React.

---

## 1. Muc tieu

### 1.1. Voice RAG

Cho phep nguoi dung upload file audio hoac ghi am bai giang, phong van, thuyet trinh, sau do:

- tu dong tach metadata audio
- transcribe speech thanh text co timestamp
- chunk transcript theo ngu nghia va thoi gian
- index transcript vao Qdrant
- hoi dap tren noi dung audio
- hien citation voi timestamp, speaker neu co
- phat lai dung doan audio lien quan

### 1.2. Video RAG

Cho phep nguoi dung upload video bai giang/hoc tap, sau do:

- tach audio va transcript
- trich xuat keyframe/slide/frame quan trong
- OCR frame/slide khi co chu
- tao caption/visual summary cho frame neu can
- canh transcript voi frame theo timestamp
- index ca text va visual evidence
- hoi dap tren video
- citation tro ve timestamp + keyframe + transcript segment
- phat video tai dung vi tri lien quan

### 1.3. Nguyen tac tich hop vao Noelys

Khong xay mot subsystem tach biet. Voice/video phai dung lai cac lop hien co:

- `Material`
- `MaterialPageDocument`
- `EvidenceBlock`
- `TextChunk`
- `RetrievedChunk`
- `HybridRetriever`
- `GraphRetriever`
- `ResponseParser`
- `EvidencePanel`

Can them kha nang temporal evidence thay vi thay doi toan bo kien truc.

---

## 2. Tham khao tu cac repo/du an lien quan

### 2.1. Audio RAG voi transcript-first

Repo `AssemblyAI-Community/rag-langchain-audio-data` trien khai RAG tren audio bang cach transcribe audio, tao embedding text, luu vao vector database, roi hoi dap tren transcript. Diem can hoc: voice RAG nen bat dau bang transcript co timestamp, khong nen dua audio raw vao retriever ngay tu dau.

Link: https://github.com/AssemblyAI-Community/rag-langchain-audio-data

### 2.2. YouTube/channel RAG

Repo `balmasi/youtube-rag` tach pipeline thanh hai phan: indexing transcript cua video va serving endpoint chat. Diem can hoc: voi video co transcript san, khong can xu ly visual truoc; co the index transcript truoc de ra MVP nhanh.

Link: https://github.com/balmasi/youtube-rag

### 2.3. Multimodal video RAG theo buoc preprocessing -> vector ingestion -> RAG

Repo `botextractai/ai-multimodal-rag-with-videos` chia video RAG thanh cac step ro rang: preprocessing video, vector store ingestion, RAG/query interface. Du an nay dung frame gan voi transcript segment va metadata timestamp. Diem can hoc: Noelys nen co job pipeline rieng cho video, gom transcript segment + representative frame + metadata.

Link: https://github.com/botextractai/ai-multimodal-rag-with-videos

### 2.4. Video RAG open-source voi Whisper + CLIP + LanceDB

Repo `indranil180/VideoRAG` dung transcript tu YouTube hoac Whisper cho local mp4, trich frame tai timestamp trung vi cua segment, tao CLIP embeddings cho frame/text, va UI cho phep play video tu timestamp dung. Diem can hoc: timestamp la citation unit quan trong nhat cua video RAG.

Link: https://github.com/indranil180/VideoRAG

### 2.5. Video-RAC cho lecture video

Video-RAC de xuat adaptive chunking bang CLIP embedding + SSIM de phat hien slide transition, chon keyframe theo entropy, canh transcript chunk voi visual chunk, va danh gia text-only/image-only/multimodal bang RAGAS. Diem can hoc: voi bai giang/slide, fixed interval chunking kem hon adaptive chunking.

Link: https://prismaticlab.github.io/Video-RAC/

### 2.6. Video-RAG cho long video

Video-RAG nhan manh cach them auxiliary text tu audio, OCR, object detection de ho tro LVLM hieu video dai ma khong can fine-tune. Diem can hoc: voi Noelys, nen uu tien bien visual/audio thanh evidence text co timestamp, roi dua vao pipeline RAG hien co.

Link: https://video-rag.github.io/

---

## 3. Chien luoc san pham

### 3.1. Giai doan 1: Voice RAG MVP

Muc tieu: upload audio, transcribe, hoi dap, citation timestamp.

Tap trung:

- `.mp3`, `.wav`, `.m4a`
- transcript segments
- Qdrant index text transcript
- evidence panel hien timestamp
- audio playback seek toi citation

Khong lam ngay:

- speaker diarization phuc tap
- emotion/audio event detection
- realtime streaming voice chat

### 3.2. Giai doan 2: Video RAG MVP

Muc tieu: upload video, tach audio, transcript, keyframes, hoi dap, citation timestamp/frame.

Tap trung:

- `.mp4`, `.mov`, `.webm`
- extract audio bang ffmpeg/moviepy
- transcribe audio
- extract keyframe moi segment
- OCR keyframe neu co text
- index transcript + OCR/caption text
- frontend co video seek/playback theo citation

Khong lam ngay:

- full multimodal embedding production-grade
- object detection moi frame
- live video stream ingestion

### 3.3. Giai doan 3: Multimodal Video RAG nang cao

Muc tieu: ket hop transcript, OCR, frame caption, CLIP/vision embeddings, Graph RAG temporal.

Tap trung:

- adaptive chunking bang SSIM/CLIP
- keyframe selection thong minh
- visual embeddings rieng
- temporal graph relations
- multimodal reranking
- RAGAS/eval cho voice/video

---

## 4. Mo hinh du lieu de xuat

### 4.1. Mo rong modality

Hien tai `Material.modality` da co huong mixed/text/image. Can them hoac chuan hoa:

```text
audio
video
mixed
```

### 4.2. Temporal metadata trong block

Them metadata vao `MaterialBlock.extra` va `EvidenceBlock.metadata` thay vi tao schema moi qua som.

De xuat fields:

```json
{
  "temporal": {
    "start_ms": 12340,
    "end_ms": 18500,
    "duration_ms": 6160
  },
  "media": {
    "media_type": "audio",
    "source_path": "data/raw/...",
    "derived_audio_path": "data/processed/...",
    "frame_path": null,
    "thumbnail_path": null
  },
  "speech": {
    "speaker": "SPEAKER_00",
    "language": "vi",
    "asr_confidence": 0.91
  }
}
```

Video block co the them:

```json
{
  "visual": {
    "frame_path": "data/processed/.../frame_000123.jpg",
    "frame_time_ms": 12340,
    "ocr_text": "...",
    "caption": "...",
    "scene_id": "scene-0004"
  }
}
```

### 4.3. Evidence block temporal fields

Trong UI va API, nen expose:

```json
{
  "page": 0,
  "block_id": "audio-seg-00012",
  "block_type": "transcript",
  "snippet_original": "...",
  "metadata": {
    "temporal": {
      "start_ms": 12340,
      "end_ms": 18500
    }
  }
}
```

Dung `page=0` hoac `page=1` cho media khong co page. Tot hon la frontend dung `metadata.temporal` khi co, khong phu thuoc page.

---

## 5. Backend implementation plan

### 5.1. Them file type support

File can sua:

- `backend/src/core/config.py`
- `config/guardrails_config.yaml`
- upload validation trong material service neu co hard-coded extension

Them extensions:

```text
mp3, wav, m4a, flac, mp4, mov, webm, mkv
```

Can gioi han size rieng cho media:

- document: 20 MB hien tai
- audio MVP: 200 MB
- video MVP: 500 MB

Neu chua muon mo size lon, bat dau 100 MB cho ca audio/video.

### 5.2. Tao media parser module

Them folder:

```text
backend/src/processing/media/
```

Files:

```text
audio_parser.py
video_parser.py
transcriber.py
frame_extractor.py
timestamp_utils.py
```

Interface nen khop voi parser hien tai:

```python
class AudioParser:
    async def parse(self, path: Path, *, material_id: str) -> ParsedDocument:
        ...

class VideoParser:
    async def parse(self, path: Path, *, material_id: str) -> ParsedDocument:
        ...
```

Output van la `ParsedDocument` voi `ParsedPage` va `ParsedBlock`.

Mapping de xuat:

- Audio:
  - `ParsedPage(page_number=1)` la toan bo audio.
  - Moi transcript segment la `ParsedBlock(block_type="transcript")`.
- Video:
  - Moi scene/chunk co the la mot `ParsedPage`.
  - Moi transcript segment, OCR frame, caption frame la `ParsedBlock`.

### 5.3. ASR/transcription

Lua chon local-first:

1. `faster-whisper` cho local ASR.
2. Whisper API/AssemblyAI la optional provider neu can cloud.

De xuat config:

```yaml
media:
  transcription_provider: "faster_whisper"
  whisper_model: "small"
  audio_segment_max_seconds: 30
  enable_speaker_diarization: false
```

ASR output can co:

```python
TranscriptSegment(
    text: str,
    start_ms: int,
    end_ms: int,
    language: str | None,
    confidence: float | None,
    speaker: str | None,
)
```

### 5.4. Audio chunking

Khong chunk transcript theo fixed token only. Nen ket hop:

- ASR segment timestamps
- punctuation/sentence boundaries
- target token count
- max duration

Quy tac MVP:

- target: 150-300 words
- max duration: 60-90 seconds
- overlap: 5-10 seconds hoac 1-2 ASR segments

Moi chunk can giu:

- `start_ms`
- `end_ms`
- list source segment ids

### 5.5. Video preprocessing

MVP:

1. Extract audio:
   - ffmpeg
   - output `.wav` or `.mp3`
2. Transcribe audio:
   - same ASR path as voice RAG
3. Extract keyframes:
   - one frame per transcript chunk at midpoint
   - save under `data/processed/{material_id}/frames/`
4. OCR frame:
   - reuse OCR pipeline if possible
5. Optional caption:
   - reuse figure captioner/VLM path if available

Advanced:

- scene detection by frame diff or SSIM
- slide transition detection
- CLIP embedding boundary detection
- keyframe entropy selection

### 5.6. Video chunking

MVP chunk unit:

```text
video_segment = transcript chunk + midpoint keyframe + optional OCR/caption
```

Content string for embedding:

```text
[Transcript]
...

[Frame OCR]
...

[Frame Caption]
...
```

Metadata:

```json
{
  "temporal": {"start_ms": 10000, "end_ms": 45000},
  "visual": {"frame_path": "...", "frame_time_ms": 27500},
  "modalities": ["audio", "video", "ocr"]
}
```

### 5.7. Embedding strategy

Phase 1:

- Use existing BGE-M3 text embeddings.
- Embed transcript + OCR + caption text.
- Store in existing Qdrant collection.

Reason: Noelys already has a strong text RAG pipeline; transcript-first gives fastest value.

Phase 2:

- Add image/frame embeddings.
- Options:
  - CLIP
  - SigLIP
  - BridgeTower-like multimodal embedding
- Add Qdrant named vector:
  - `dense_text`
  - `dense_visual`
  - existing `sparse`

Need migration plan because current Qdrant config uses named dense vector `dense` and sparse vector `bge_m3_sparse`.

### 5.8. Retrieval strategy

Voice:

```text
query -> text embedding -> transcript chunks -> rerank -> answer
```

Video MVP:

```text
query -> text embedding -> transcript/OCR/caption chunks -> rerank -> answer + timestamp/frame citation
```

Video advanced:

```text
query -> route
  - text-heavy query: transcript/OCR retrieval
  - visual query: visual embedding retrieval + caption/OCR retrieval
  - relation query: Graph RAG over temporal entities
merge -> rerank -> answer
```

### 5.9. Graph RAG extension

Add temporal relations:

```text
entity:dropout --mentioned_in_segment--> segment:00:12-00:45
segment:00:12-00:45 --has_keyframe--> frame:00:28
entity:overfitting --co_occurs_in_segment--> entity:dropout
```

Use cases:

- "Trong video, dropout lien quan gi den overfitting?"
- "Khuc nao thay noi ve regularization?"
- "Sau khi giai thich overfitting, thay noi gi tiep?"

Need relation types:

```text
mentioned_in_segment
co_occurs_in_segment
shown_in_frame
explained_before
explained_after
has_keyframe
has_transcript
has_ocr
```

### 5.10. API changes

Existing upload endpoints can remain:

- `/api/v1/materials/upload`
- `/api/v1/materials/batch_upload`

Need enhance responses/status:

- expose media duration
- expose processing stage:
  - extracting_audio
  - transcribing
  - extracting_frames
  - ocr_frames
  - indexing_media

Optional new endpoint:

```text
GET /api/v1/materials/{material_id}/media
```

Response:

```json
{
  "material_id": "...",
  "media_type": "video",
  "duration_ms": 123456,
  "stream_url": "/api/v1/materials/{id}/raw"
}
```

Could reuse `/materials/{material_id}/raw` for playback if range requests work. If not, implement range support.

---

## 6. Frontend implementation plan

### 6.1. Upload UI

Update accepted file types:

- audio files
- video files

Show media-specific upload note:

- "Audio/video may take longer because Noelys transcribes and indexes timestamps."

### 6.2. Source list

Add icons:

- audio file
- video file

Show status badges:

- Transcribing
- Extracting frames
- Indexing media
- Ready

### 6.3. Evidence panel

For audio/video citation:

- show timestamp range: `00:12 - 00:45`
- show play button
- show transcript snippet
- for video show keyframe thumbnail
- clicking citation seeks media player

Data from citation:

```ts
metadata.temporal.start_ms
metadata.temporal.end_ms
metadata.visual.frame_path
```

### 6.4. Media viewer

Add a reusable component:

```text
MediaEvidencePlayer.tsx
```

Responsibilities:

- audio/video playback
- seek to timestamp
- show transcript synced with playback
- show selected evidence highlight

### 6.5. Workspace interaction

When user clicks citation:

1. Select citation as today.
2. If citation has `metadata.temporal`, open media viewer mode.
3. Seek to `start_ms`.
4. Highlight transcript segment.

---

## 7. Storage layout

Suggested layout:

```text
data/
  raw/
    {owner_id}/
      {material_id}/original.mp4
  processed/
    {owner_id}/
      {material_id}/
        audio/
          extracted.wav
        transcript/
          transcript.json
          transcript.vtt
        frames/
          frame_000001.jpg
          frame_000002.jpg
        thumbnails/
          thumb_000001.jpg
        ocr/
          frame_ocr.json
```

Transcript JSON:

```json
{
  "segments": [
    {
      "segment_id": "seg-000001",
      "start_ms": 0,
      "end_ms": 5320,
      "text": "...",
      "speaker": null,
      "confidence": 0.91
    }
  ]
}
```

---

## 8. Dependencies

### 8.1. MVP dependencies

Backend:

```text
ffmpeg-python
moviepy
faster-whisper
webvtt-py
opencv-python
```

System:

```text
ffmpeg
```

Optional:

```text
yt-dlp
```

Use `yt-dlp` only if supporting URL ingestion. For first implementation, avoid URL ingestion and only support uploaded files.

### 8.2. Advanced dependencies

```text
scenedetect
transformers
open_clip_torch
scikit-image
```

These should not be added in MVP unless needed because they increase install/runtime complexity.

---

## 9. Configuration proposal

Add to `config/model_config.yaml`:

```yaml
media:
  enabled: true
  ffmpeg_path: "ffmpeg"
  max_audio_size_mb: 200
  max_video_size_mb: 500

  transcription:
    provider: "faster_whisper"
    model_name: "small"
    device: "cpu"
    compute_type: "int8"
    language:

  video:
    keyframe_strategy: "midpoint"
    frame_interval_seconds: 30
    enable_frame_ocr: true
    enable_frame_captioning: false
    max_frames_per_video: 300
```

Add env overrides:

```text
AGENTBOOK_MEDIA_ENABLED=true
AGENTBOOK_TRANSCRIPTION_PROVIDER=faster_whisper
AGENTBOOK_WHISPER_MODEL=small
AGENTBOOK_MEDIA_MAX_AUDIO_SIZE_MB=200
AGENTBOOK_MEDIA_MAX_VIDEO_SIZE_MB=500
```

---

## 10. Step-by-step roadmap

### Phase 0 - Design hardening

Deliverables:

- final schema decision for temporal metadata
- decide supported file extensions
- decide ASR provider
- decide whether to use eager processing or Celery for media

Acceptance criteria:

- one design doc approved
- no breaking changes to existing document upload

### Phase 1 - Voice RAG MVP

Backend tasks:

1. Add audio extensions to validation.
2. Add `AudioParser`.
3. Add `Transcriber` abstraction.
4. Implement `FasterWhisperTranscriber`.
5. Convert transcript segments into `ParsedBlock`.
6. Preserve `start_ms/end_ms` in block metadata.
7. Ensure chunking keeps temporal metadata.
8. Ensure citations return temporal metadata.
9. Add tests for audio parse -> chunks -> citations.

Frontend tasks:

1. Allow audio uploads.
2. Show audio source icon.
3. Add timestamp rendering in EvidencePanel.
4. Add audio player and seek-to-citation.

Acceptance criteria:

- upload `.mp3` or `.wav`
- ask "noi dung chinh cua audio la gi"
- answer includes citations
- clicking citation plays audio at correct timestamp

### Phase 2 - Video RAG MVP

Backend tasks:

1. Add video extensions to validation.
2. Add `VideoParser`.
3. Extract audio with ffmpeg.
4. Reuse transcriber.
5. Extract midpoint keyframe for each transcript chunk.
6. Run OCR on keyframes when enabled.
7. Build video segment blocks with transcript + OCR/caption text.
8. Store frame paths in metadata.
9. Return video timestamp/frame evidence to frontend.

Frontend tasks:

1. Allow video uploads.
2. Show video source icon.
3. Add video player.
4. Add keyframe thumbnail in evidence.
5. Seek video when citation selected.

Acceptance criteria:

- upload `.mp4`
- ask "doan nao noi ve dropout"
- answer cites segment
- evidence panel shows timestamp + thumbnail
- clicking citation seeks video to segment

### Phase 3 - Better video chunking

Backend tasks:

1. Implement frame diff/SSIM scene detection.
2. Optional CLIP boundary detection.
3. Align transcript chunks with visual chunks.
4. Select first/mid/last or entropy keyframes per chunk.
5. Compare fixed interval vs adaptive chunking.

Acceptance criteria:

- slide lecture videos retrieve more accurate segments than fixed midpoint extraction
- evaluation shows better context relevance

### Phase 4 - Multimodal retrieval

Backend tasks:

1. Add visual embedding provider interface.
2. Add Qdrant named vector for visual embeddings.
3. Add multimodal retrieval route.
4. Merge text, OCR, caption, and visual hits.
5. Add multimodal reranking or heuristic scoring.

Acceptance criteria:

- visual queries such as "slide co bieu do accuracy" return frame evidence even if transcript does not mention all text
- citations include frame thumbnail and timestamp

### Phase 5 - Temporal Graph RAG

Backend tasks:

1. Extract entities from transcript/caption/OCR.
2. Create temporal relations.
3. Add graph retriever support for segment/frame nodes.
4. Add reasoning path descriptions for temporal graph traversal.

Acceptance criteria:

- relationship query over video includes `traverse`
- graph panel can show entity -> segment -> frame/source

### Phase 6 - Evaluation and quality gate

Tasks:

1. Build small eval set:
   - 5 audio files
   - 5 lecture videos
   - 30 QA pairs
2. Measure:
   - retrieval hit rate
   - timestamp accuracy
   - citation faithfulness
   - answer language compliance
   - latency
3. Add regression tests for:
   - transcript parsing
   - timestamp preservation
   - evidence selection
   - frontend citation seek behavior

Acceptance criteria:

- timestamp hit within correct segment for at least 80% MVP eval questions
- citation block selected correctly for common voice/video questions

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ASR is slow on CPU | Upload/indexing can hang | Use eager only for short files; Celery for long media; expose progress stages |
| Whisper hallucination/noisy transcript | Bad answers | Store ASR confidence; add low-confidence warning; allow transcript preview |
| Video files too large | Storage/processing pressure | Add size limits and duration limits |
| Fixed interval chunks miss slide boundaries | Weak retrieval | Phase 3 adaptive chunking |
| Visual embeddings add complexity | More infra/migrations | Start transcript-first; add visual vector later |
| Frontend playback from raw endpoint lacks range support | Seek may fail | Implement HTTP range support for media raw endpoint |
| Mongo/Qdrant schema migration | Existing data compatibility | Use metadata-first approach before schema hard changes |

---

## 12. Recommended first implementation slice

Lam theo thu tu nay de co ket qua nhanh nhat:

1. Add audio file extensions.
2. Implement `Transcriber` + `AudioParser`.
3. Save transcript segments as `ParsedBlock(block_type="transcript")`.
4. Preserve temporal metadata through chunking and evidence.
5. Render timestamp in EvidencePanel.
6. Add audio player seek-to-citation.
7. Add video parser by reusing audio path and extracting midpoint frames.

Khong nen lam CLIP/multimodal embeddings ngay trong slice dau. Noelys da co Hybrid RAG text tot; transcript-first se cho gia tri thuc te nhanh hon va it pha vo kien truc hien tai.

---

## 13. File/code areas expected to change

Backend:

```text
backend/src/core/config.py
backend/src/models/common.py
backend/src/models/material.py
backend/src/processing/types.py
backend/src/processing/media/audio_parser.py
backend/src/processing/media/video_parser.py
backend/src/processing/media/transcriber.py
backend/src/processing/media/frame_extractor.py
backend/src/services/parse_index_pipeline.py
backend/src/services/material_service.py
backend/src/inference/response_parser.py
backend/src/schemas/evidence.py
backend/src/schemas/material.py
```

Frontend:

```text
frontend/src/api/client.ts
frontend/src/components/EvidencePanel.tsx
frontend/src/components/workspace/ChatPanel.tsx
frontend/src/components/MediaEvidencePlayer.tsx
frontend/src/components/SourceList.tsx
```

Tests:

```text
backend/tests/test_processing/test_audio_parser.py
backend/tests/test_processing/test_video_parser.py
backend/tests/test_inference/test_temporal_citations.py
frontend/src/**/*.test.tsx
```

---

## 14. Open decisions

1. Chon ASR mac dinh: `faster-whisper` local hay cloud provider optional?
2. Co bat speaker diarization trong MVP khong?
3. Video max duration la bao nhieu cho local dev?
4. Co can support YouTube URL ingestion hay chi upload file?
5. Co chap nhan them FFmpeg la system dependency bat buoc khong?
6. Temporal evidence nen expose trong `metadata` hay them field typed rieng vao schema?
7. Co can migration Qdrant named vector de them visual embeddings ngay tu dau khong?

Khuyen nghi:

- MVP: local `faster-whisper`, no diarization, upload only, FFmpeg required, metadata-first, no visual vector migration.

---

## 15. References

- AssemblyAI audio RAG repo: https://github.com/AssemblyAI-Community/rag-langchain-audio-data
- YouTube transcript RAG repo: https://github.com/balmasi/youtube-rag
- Multimodal video RAG repo: https://github.com/botextractai/ai-multimodal-rag-with-videos
- Open-source VideoRAG repo: https://github.com/indranil180/VideoRAG
- Video-RAC lecture video chunking: https://prismaticlab.github.io/Video-RAC/
- Video-RAG long video project: https://video-rag.github.io/
