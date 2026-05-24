import { useRef, useState } from "react";
import { Pause, Play, Volume2 } from "lucide-react";
import { API_V1_BASE_URL, EvidenceBlock } from "../api/client";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Render a list of audio transcript segments with clickable [MM:SS] timestamps.
 *  Click a row → seek the embedded `<audio>` to that timestamp and play just that segment.
 *
 *  Reads audio metadata from EvidenceBlock fields: `audio_start_seconds`,
 *  `audio_end_seconds`, `audio_file`, `material_id`. Renders nothing when the
 *  citation has no audio blocks.
 */
export function AudioSegmentList({ evidenceBlocks, ownerId }: {
  evidenceBlocks: EvidenceBlock[];
  ownerId: string;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingIdx, setPlayingIdx] = useState<number | null>(null);

  const audioSegments = evidenceBlocks.filter(
    (b) => b.audio_start_seconds != null && b.audio_end_seconds != null && b.material_id,
  );
  if (audioSegments.length === 0) return null;

  const materialId = audioSegments[0].material_id;
  const fileName = audioSegments[0].audio_file || audioSegments[0].doc_name || "audio";
  const audioUrl = `${API_V1_BASE_URL}/materials/${materialId}/raw?owner_id=${encodeURIComponent(ownerId)}`;

  function playSegment(idx: number) {
    const seg = audioSegments[idx];
    const audio = audioRef.current;
    if (!audio || seg.audio_start_seconds == null) return;
    const start = seg.audio_start_seconds;
    const end = seg.audio_end_seconds ?? start;
    if (playingIdx === idx) {
      audio.pause();
      setPlayingIdx(null);
      return;
    }
    audio.currentTime = start;
    audio.play().then(() => setPlayingIdx(idx)).catch(() => setPlayingIdx(null));
    const checkEnd = () => {
      if (audio.currentTime >= end) {
        audio.pause();
        setPlayingIdx(null);
        audio.removeEventListener("timeupdate", checkEnd);
      }
    };
    audio.addEventListener("timeupdate", checkEnd);
  }

  return (
    <div className="my-2 rounded-lg border border-outline bg-surface-low shadow-sm overflow-hidden">
      <audio ref={audioRef} src={audioUrl} preload="metadata" />
      <div className="px-3 py-2 border-b border-outline/50 bg-primary/5 flex items-center gap-2">
        <Volume2 size={12} className="text-primary" />
        <span className="text-[11px] font-semibold text-text truncate" title={fileName}>
          {fileName}
        </span>
        <span className="ml-auto text-[10px] text-muted">{audioSegments.length} đoạn</span>
      </div>
      <ul className="divide-y divide-outline/30 max-h-72 overflow-y-auto">
        {audioSegments.map((seg, idx) => {
          const isPlaying = playingIdx === idx;
          return (
            <li key={idx}>
              <button
                type="button"
                onClick={() => playSegment(idx)}
                className={`w-full flex items-start gap-2.5 px-3 py-2 text-left transition hover:bg-primary/5 ${
                  isPlaying ? "bg-primary/10" : ""
                }`}
              >
                <span className={`shrink-0 mt-0.5 flex items-center justify-center h-5 w-5 rounded-full ${
                  isPlaying ? "bg-primary text-white" : "bg-outline/40 text-muted"
                }`}>
                  {isPlaying ? <Pause size={9} fill="currentColor" /> : <Play size={9} fill="currentColor" className="ml-0.5" />}
                </span>
                <span className="shrink-0 text-[10px] font-mono font-bold text-primary tabular-nums">
                  {formatTime(seg.audio_start_seconds!)}
                </span>
                <span className="flex-1 text-[11px] leading-relaxed text-text">
                  {seg.snippet_original}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
