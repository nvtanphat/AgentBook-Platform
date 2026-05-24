import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, FileText, Image, ImagePlus, Loader2, Network, Send, Table2, Trash2, Library, X, Workflow } from "lucide-react";
import { API_V1_BASE_URL, Citation, QueryResponse, SentenceCoverageReport, askQuestionStream, askQuestionWithImage } from "../../api/client";
import { useWorkspace } from "../../state/workspace";
import { StudioTab } from "../../pages/WorkspacePage";
import MarkdownRenderer from "../MarkdownRenderer";
import ReasoningTrace from "../ReasoningTrace";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
};

const CHAT_STORAGE_PREFIX = "prism.chat.v1";

function chatStorageKey(ownerId: string, collectionId: string) {
  return `${CHAT_STORAGE_PREFIX}:${ownerId}:${collectionId || "no_collection"}`;
}

function getIntroMessage(language: string): string {
  if (language === "vi") return "Xin chào! Tôi là Noelys. Hãy tải tài liệu lên bên trái để bắt đầu hỏi đáp có căn cứ, hoặc hỏi tôi bất cứ điều gì.";
  if (language === "zh") return "你好！我是 Noelys 助手。请在左侧上传文档以开始有依据的问答。";
  if (language === "ja") return "こんにちは！Noelysアシスタントです。左側にドキュメントをアップロードして始めてください。";
  if (language === "ko") return "안녕하세요! Noelys 어시스턴트입니다. 왼쪽에 문서를 업로드하여 시작하세요.";
  return "Welcome to Noelys! Upload some sources on the left to start grounded Q&A, or ask me anything.";
}

function friendlyRefusal(reason: string | null | undefined): string {
  if (!reason) return "Tôi không thể trả lời câu hỏi này dựa trên tài liệu hiện có.";
  if (reason.includes("confidence") || reason.includes("low")) return "Tôi chưa đủ tự tin để trả lời câu hỏi này. Bằng chứng tìm được quá yếu — hãy thử diễn đạt lại hoặc kiểm tra lại nguồn.";
  if (reason.includes("no relevant") || reason.includes("evidence")) return "Tôi không tìm thấy thông tin liên quan trong tài liệu của bạn.";
  if (reason.includes("scope") || reason.includes("missing")) return "Vui lòng chọn một collection hoặc thêm tài liệu trước khi đặt câu hỏi.";
  return `Không thể trả lời: ${reason}`;
}

function friendlyError(err: unknown): string {
  const raw = err instanceof Error ? err.message : "Không thể hoàn tất thao tác.";
  if (/network|fetch|timeout|failed to fetch/i.test(raw)) return "Không kết nối được server. Kiểm tra backend rồi thử lại.";
  if (/401|403|unauthorized|forbidden/i.test(raw)) return "Bạn không có quyền truy cập collection này. Kiểm tra owner hoặc phiên đăng nhập.";
  if (/500|internal server/i.test(raw)) return "Server gặp lỗi khi xử lý. Thử lại sau hoặc giảm phạm vi tài liệu.";
  return raw;
}

function agentStepLabel(name: string): string {
  const labels: Record<string, string> = {
    plan_query: "Đang lập kế hoạch truy xuất...",
    retrieve_text: "Đang tìm bằng chứng trong tài liệu...",
    retrieve_multi_query: "Đang tìm bằng chứng với nhiều cách diễn đạt...",
    retrieve_per_source: "Đang quét từng nguồn đã chọn...",
    retrieve_evidence: "Đang điều phối multi-tool retrieval...",
    trace_graph: "Đang truy vết quan hệ trên Knowledge Graph...",
    verify_coverage: "Đang kiểm tra độ phủ nguồn...",
    repair_retrieval: "Đang bổ sung bằng chứng còn thiếu...",
    verify_evidence_quality: "Đang kiểm tra chất lượng bằng chứng...",
    crag_triage: "Đang triage bằng chứng (CORRECT/AMBIGUOUS/INCORRECT)...",
    rerank_evidence: "Đang xếp hạng bằng chứng phù hợp...",
    synthesize_answer: "Đang tổng hợp câu trả lời...",
    repair_answer: "Đang sửa câu trả lời theo bằng chứng...",
    verify_claims: "Đang kiểm chứng câu trả lời (NLI guardrails)...",
    critic_review: "Critic agent đang review câu trả lời...",
    critic_refined_synthesis: "Đang viết lại câu trả lời với bằng chứng bổ sung...",
  };
  return labels[name] ?? `Đang xử lý: ${name.replace(/_/g, " ")}`;
}

function agentStepRole(name: string): string {
  if (name === "plan_query") return "PLANNER";
  if (name.startsWith("retrieve") || name === "trace_graph" || name === "repair_retrieval") return "DIRECTOR";
  if (name === "crag_triage") return "CRAG CRITIC";
  if (name === "rerank_evidence" || name === "verify_evidence_quality" || name === "verify_coverage") return "RERANKER";
  if (name === "synthesize_answer" || name === "repair_answer" || name === "critic_refined_synthesis") return "SYNTHESIZER";
  if (name === "verify_claims") return "GUARDRAILS";
  if (name === "critic_review") return "CRITIC";
  return "AGENT";
}

// Color theme per agent role — clean visual differentiation, no emoji noise
function agentRoleTheme(role: string): { dot: string; chip: string } {
  const themes: Record<string, { dot: string; chip: string }> = {
    PLANNER:      { dot: "bg-indigo-400",  chip: "border-indigo-200 bg-indigo-50 text-indigo-700" },
    DIRECTOR:     { dot: "bg-sky-400",     chip: "border-sky-200 bg-sky-50 text-sky-700" },
    "CRAG CRITIC":{ dot: "bg-emerald-400", chip: "border-emerald-200 bg-emerald-50 text-emerald-700" },
    RERANKER:     { dot: "bg-violet-400",  chip: "border-violet-200 bg-violet-50 text-violet-700" },
    SYNTHESIZER:  { dot: "bg-amber-400",   chip: "border-amber-200 bg-amber-50 text-amber-700" },
    GUARDRAILS:   { dot: "bg-rose-400",    chip: "border-rose-200 bg-rose-50 text-rose-700" },
    CRITIC:       { dot: "bg-slate-400",   chip: "border-slate-200 bg-slate-50 text-slate-700" },
    AGENT:        { dot: "bg-slate-300",   chip: "border-slate-200 bg-slate-50 text-muted" },
  };
  return themes[role] ?? themes.AGENT;
}

function agentTraceStepLabel(name: string): string {
  return agentStepLabel(name)
    .replace(/^Đang\s+/i, "")
    .replace(/\s+đang\s+/gi, " ")
    .replace(/\.\.\.$/, "");
}

function normalizeMessage(value: string) {
  return value.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").trim();
}

function casualReply(message: string, hasScope: boolean): string | null {
  const text = normalizeMessage(message);
  if (/^(hi|hello|hey|alo|chao|xin chao|test|ping)\b/.test(text)) {
    return hasScope
      ? "Xin chào! Tôi sẵn sàng trả lời câu hỏi về collection của bạn."
      : "Xin chào! Hãy thêm tài liệu để bắt đầu hỏi đáp có căn cứ.";
  }
  if (text.includes("who are you") || text.includes("ban la ai")) {
    return "Tôi là Noelys, trợ lý AI chuyên đọc hiểu tài liệu, truy xuất bằng chứng và hỗ trợ học tập.";
  }
  return null;
}

// ─── Confidence badge ─────────────────────────────────────────────────────────

// ─── Sentence-level Evidence Coverage badge ──────────────────────────────────
// Surface artefact of the SLEC gate: shows what fraction of the answer's
// sentences had evidence support, with click-to-expand per-sentence breakdown.

function statusColor(status: "supported" | "partial" | "unsupported") {
  if (status === "supported") return { dot: "#10b981", text: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200", label: "Có bằng chứng" };
  if (status === "partial") return { dot: "#f59e0b", text: "text-amber-700", bg: "bg-amber-50", border: "border-amber-200", label: "Bằng chứng yếu" };
  return { dot: "#ef4444", text: "text-red-700", bg: "bg-red-50", border: "border-red-200", label: "Không có bằng chứng" };
}

function EvidenceCoverageBadge({ report }: { report: SentenceCoverageReport }) {
  const [open, setOpen] = useState(false);
  if (!report.enabled || report.total_sentences === 0) return null;
  const pct = Math.round(report.coverage_ratio * 100);
  const cls = pct >= 80 ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : pct >= 50 ? "bg-amber-50 text-amber-700 border-amber-200"
    : "bg-red-50 text-red-700 border-red-200";
  const dot = pct >= 80 ? "#10b981" : pct >= 50 ? "#f59e0b" : "#ef4444";
  const droppedHint = report.dropped_count > 0 ? ` · loại ${report.dropped_count} câu không có bằng chứng` : "";

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={`Bằng chứng phủ ${pct}% · ${report.supported_count}/${report.total_sentences} câu được hỗ trợ${droppedHint}\nClick để xem chi tiết từng câu.`}
        className={`inline-flex cursor-pointer items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold transition hover:opacity-90 ${cls}`}
      >
        <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: dot }} />
        Bằng chứng phủ {pct}% · {report.supported_count}/{report.total_sentences}
        {report.dropped_count > 0 && <span className="ml-1 opacity-70">· loại {report.dropped_count}</span>}
        <ChevronDown size={10} className={`ml-0.5 transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && report.sentences.length > 0 && (
        <ul className="mt-1.5 space-y-1 rounded-lg border border-outline/30 bg-surface-low/60 p-2">
          {report.sentences.map((s) => {
            const c = statusColor(s.status);
            return (
              <li key={s.index} className="flex items-start gap-2 text-[11px]">
                <span
                  className={`mt-0.5 inline-flex shrink-0 items-center gap-1 rounded-full border px-1.5 py-0.5 font-bold ${c.bg} ${c.text} ${c.border}`}
                  title={`${c.label} · score ${(s.score * 100).toFixed(0)}%`}
                >
                  <span className="inline-block h-1 w-1 rounded-full" style={{ background: c.dot }} />
                  {(s.score * 100).toFixed(0)}%
                </span>
                <span className="leading-snug text-muted">{s.text}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 70 ? "bg-emerald-50 text-emerald-700 border-emerald-200" : pct >= 40 ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-red-50 text-red-700 border-red-200";
  const label = pct >= 70 ? "Tin cậy cao" : pct >= 40 ? "Tin cậy trung bình" : "Tin cậy thấp";
  return (
    <span
      title={`${label} (${pct}%) — Điểm tin cậy dựa trên bằng chứng tìm được.\n≥70%: tốt · 40–70%: trung bình · <40%: cần kiểm tra lại`}
      className={`confidence-badge inline-flex cursor-help items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold ${cls}`}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: pct >= 70 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#ef4444' }} />
      {label} · {pct}%
    </span>
  );
}

// ─── Message content (full markdown with citation refs) ───────────────────────

function MessageContent({ content, citations, onCitationClick }: {
  content: string;
  citations: Citation[];
  onCitationClick: (idx: number) => void;
}) {
  return (
    <MarkdownRenderer
      text={content}
      onCitationClick={(ref) => {
        if (ref >= 0 && ref < citations.length) onCitationClick(ref);
      }}
    />
  );
}

function AgentTraceBody({ trace }: { trace: NonNullable<QueryResponse["agent_trace"]> }) {
  const totalMs = trace.steps.reduce((acc, s) => acc + (s.duration_ms ?? 0), 0);
  const roleCounts = trace.steps.reduce<Record<string, number>>((acc, s) => {
    const role = agentStepRole(s.name);
    acc[role] = (acc[role] ?? 0) + 1;
    return acc;
  }, {});
  const uniqueRoles = Object.keys(roleCounts);

  return (
    <div className="rounded-lg border border-outline/60 bg-surface-low/60">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-outline/40">
        <Workflow size={13} className="text-primary shrink-0" />
        <span className="text-[11px] font-bold uppercase tracking-wider text-text">
          Luồng xử lý agent
        </span>
        <span className="text-[10px] font-medium text-muted truncate">
          · {trace.steps.length} bước · {uniqueRoles.length} agent{totalMs > 0 ? ` · ${(totalMs / 1000).toFixed(1)}s` : ''}
        </span>
      </div>
      <div className="px-3 py-3">
          {/* Plan + role legend row */}
          <div className="mb-3 flex flex-wrap items-center gap-1.5 text-[10px]">
            <span className="rounded border border-primary/30 bg-primary/5 px-2 py-0.5 font-bold text-primary uppercase tracking-wide">
              {trace.plan_type}
            </span>
            {uniqueRoles.map((role) => {
              const theme = agentRoleTheme(role);
              return (
                <span key={role} className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 font-semibold uppercase tracking-wide text-[9px] ${theme.chip}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${theme.dot}`} />
                  {role}{roleCounts[role] > 1 ? ` ×${roleCounts[role]}` : ''}
                </span>
              );
            })}
            {trace.repair_attempted && (
              <span className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-blue-700 font-semibold uppercase tracking-wide text-[9px]">
                self-repair
              </span>
            )}
          </div>

          {/* Steps timeline */}
          <div className="space-y-1">
            {trace.steps.map((step, index) => {
              const role = agentStepRole(step.name);
              const theme = agentRoleTheme(role);
              const isLastStep = index === trace.steps.length - 1;
              const statusColor =
                step.status === "completed" ? "border-emerald-200" :
                step.status === "failed" ? "border-red-200" :
                step.status === "skipped" ? "border-slate-200 opacity-60" :
                "border-amber-200";
              return (
                <div key={`${step.name}-${index}`} className="relative pl-6">
                  {/* Timeline rail */}
                  {!isLastStep && (
                    <span className="absolute left-[10px] top-4 h-[calc(100%-2px)] w-px bg-outline/40" aria-hidden />
                  )}
                  {/* Role dot */}
                  <span
                    className={`absolute left-[6px] top-[7px] h-2.5 w-2.5 rounded-full ring-2 ring-surface-low ${theme.dot}`}
                    title={role}
                  />
                  <div className={`rounded border bg-white px-2.5 py-1.5 ${statusColor}`}>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${theme.chip}`}>
                        {role}
                      </span>
                      <span className="text-[11px] font-medium text-text">
                        {agentTraceStepLabel(step.name)}
                      </span>
                      <span className="ml-auto flex items-center gap-1.5 shrink-0">
                        {step.duration_ms != null && step.duration_ms > 0 && (
                          <span className="text-[10px] font-semibold text-muted tabular-nums">
                            {step.duration_ms < 1000 ? `${step.duration_ms}ms` : `${(step.duration_ms / 1000).toFixed(1)}s`}
                          </span>
                        )}
                        {step.status === "completed" ? (
                          <CheckCircle2 size={11} className="text-emerald-500" />
                        ) : step.status === "failed" ? (
                          <AlertCircle size={11} className="text-red-500" />
                        ) : step.status === "skipped" ? (
                          <ChevronDown size={11} className="text-muted" />
                        ) : (
                          <Loader2 size={11} className="text-amber-500 animate-spin" />
                        )}
                      </span>
                    </div>

                    {/* Detail row — text only, no emoji */}
                    {((step.sources_requested != null && step.sources_requested > 0 && role !== "PLANNER")
                      || (step.evidence_count != null && step.evidence_count > 0)
                      || step.warning || typeof step.metadata?.correct === "number"
                      || typeof step.metadata?.sub_questions_covered === "number") && (
                      <div className="mt-1 flex flex-wrap gap-1.5 text-[10px] text-muted">
                        {step.sources_requested != null && step.sources_requested > 0 && role !== "PLANNER" && (
                          <span>{step.sources_covered ?? 0}/{step.sources_requested} nguồn</span>
                        )}
                        {step.evidence_count != null && step.evidence_count > 0 && (
                          <span>· {step.evidence_count} {role === "GUARDRAILS" ? "câu kiểm tra" : "bằng chứng"}</span>
                        )}
                        {typeof step.metadata?.sub_questions_covered === "number" && typeof step.metadata?.sub_questions_requested === "number" && (
                          <span>· {String(step.metadata.sub_questions_covered)}/{String(step.metadata.sub_questions_requested)} sub-Q</span>
                        )}
                        {/* CRAG triage compact */}
                        {typeof step.metadata?.correct === "number" && (
                          <span className="text-emerald-700 font-semibold">
                            {String(step.metadata.correct)} correct
                            {Number(step.metadata.ambiguous ?? 0) > 0 && <span className="text-amber-700">, {String(step.metadata.ambiguous)} amb</span>}
                            {Number(step.metadata.incorrect ?? 0) > 0 && <span className="text-red-700">, {String(step.metadata.incorrect)} wrong</span>}
                          </span>
                        )}
                        {step.warning && (
                          <span className="text-amber-700">· {step.warning}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Verification summary — minimal, no emoji */}
          {trace.verification && (
            <div className="mt-3 rounded border border-outline/40 bg-white px-3 py-2 text-[10px]">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-bold uppercase tracking-wider text-muted">Guardrails</span>
                <span className={`rounded px-2 py-0.5 font-bold uppercase tracking-wide ${
                  trace.verification.verdict === "supported" ? "bg-emerald-50 text-emerald-700 border border-emerald-200" :
                  trace.verification.verdict === "contradicted" ? "bg-red-50 text-red-700 border border-red-200" :
                  "bg-amber-50 text-amber-700 border border-amber-200"
                }`}>
                  {trace.verification.verdict}
                </span>
                <span className="font-semibold tabular-nums text-muted">
                  {Math.round(trace.verification.confidence * 100)}%
                </span>
                {trace.verification.repair_attempted && (
                  <span className="text-blue-700 font-medium">· đã tự sửa</span>
                )}
              </div>
              {trace.verification.warning && (
                <div className="mt-1 text-amber-700">{trace.verification.warning}</div>
              )}
              {Boolean(trace.verification.unsupported_sentence_count || trace.verification.invalid_citation_count) && (
                <div className="mt-1 text-amber-700">
                  {trace.verification.unsupported_sentence_count ?? 0} câu thiếu citation, {trace.verification.invalid_citation_count ?? 0} citation sai.
                </div>
              )}
            </div>
          )}
        </div>
    </div>
  );
}

function AnswerMeta({ response, onTraceGraph }: { response: QueryResponse; onTraceGraph?: () => void }) {
  const [open, setOpen] = useState(false);
  const coverage = response.coverage;
  const trace = response.agent_trace;
  const verification = trace?.verification;
  const completeCoverage = coverage ? coverage.covered_count >= coverage.requested_count : true;
  const verified = verification?.verdict === "supported";
  const hasReasoning = Boolean(response.reasoning_path && response.reasoning_path.length > 0);
  const hasCitationWarn = Boolean(verification?.invalid_citation_count || verification?.unsupported_sentence_count);
  const hasGraphButton = Boolean(onTraceGraph && response.citations.length > 0);
  const hasExtras = Boolean(trace?.repair_attempted || verification?.repair_attempted || hasCitationWarn || trace);
  const hasExpandable = hasReasoning || Boolean(trace) || hasGraphButton || hasExtras;

  return (
    <div className="mt-2 space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <ConfidenceBadge value={response.confidence} />
        {coverage && coverage.requested_count > 0 && (
          <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold ${
            completeCoverage ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"
          }`}>
            {completeCoverage ? <CheckCircle2 size={10} /> : <AlertTriangle size={10} />}
            Độ phủ {coverage.covered_count}/{coverage.requested_count}
          </span>
        )}
        {verification && (
          <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-bold ${
            verified ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"
          }`}>
            {verified ? <CheckCircle2 size={10} /> : <AlertTriangle size={10} />}
            {verified ? "Đã kiểm chứng" : "Cần kiểm tra"}
          </span>
        )}
        {response.sentence_coverage && (
          <EvidenceCoverageBadge report={response.sentence_coverage} />
        )}
        {hasExpandable && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="ml-auto inline-flex items-center gap-1 rounded-full border border-outline/50 bg-white px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
            title={open ? "Ẩn chi tiết" : "Xem chi tiết truy xuất, agent trace và graph"}
          >
            Chi tiết
            {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-2">
          {hasReasoning && (
            <ReasoningTrace steps={response.reasoning_path!} onStepHover={() => undefined} />
          )}
          {hasExtras && (
            <div className="flex flex-wrap items-center gap-1.5">
              {trace?.repair_attempted && (
                <span className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-bold text-blue-700">
                  Đã bổ sung truy xuất
                </span>
              )}
              {verification?.repair_attempted && (
                <span className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-bold text-blue-700">
                  Đã sửa câu trả lời
                </span>
              )}
              {hasCitationWarn && (
                <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold text-amber-700">
                  <AlertTriangle size={10} />
                  Cần kiểm tra citation
                </span>
              )}
              {trace && (
                <span className="inline-flex items-center rounded-full border border-outline bg-slate-50 px-2 py-0.5 text-[10px] font-bold text-muted">
                  Kế hoạch: {trace.plan_type.replace(/_/g, " ")}
                </span>
              )}
            </div>
          )}
          {trace && <AgentTraceBody trace={trace} />}
          {hasGraphButton && <TraceGraphAction onTraceGraph={onTraceGraph} />}
        </div>
      )}
    </div>
  );
}

function TraceGraphAction({ onTraceGraph }: { onTraceGraph?: () => void }) {
  if (!onTraceGraph) return null;
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={onTraceGraph}
        className="inline-flex items-center gap-1.5 rounded-full border border-primary/25 bg-primary/5 px-3 py-1.5 text-[11px] font-bold text-primary transition hover:border-primary/45 hover:bg-primary/10"
        title="Mở Knowledge Graph để truy vết quan hệ và bằng chứng của câu trả lời này"
      >
        <Network size={12} />
        Kiểm chứng bằng Graph
      </button>
    </div>
  );
}

// ─── Citation footer ──────────────────────────────────────────────────────────

function citationFileIcon(name: string) {
  if (/\.(png|jpe?g)$/i.test(name)) return <Image size={10} className="text-purple-400 shrink-0" />;
  if (/\.pdf$/i.test(name)) return <FileText size={10} className="text-red-400 shrink-0" />;
  if (/\.docx?$/i.test(name)) return <FileText size={10} className="text-blue-400 shrink-0" />;
  if (/\.pptx?$/i.test(name)) return <FileText size={10} className="text-amber-400 shrink-0" />;
  if (/\.(csv|xlsx)$/i.test(name)) return <Table2 size={10} className="text-emerald-500 shrink-0" />;
  return <FileText size={10} className="text-muted shrink-0" />;
}

function isImageCitation(c: Citation): boolean {
  const name = (c.doc_name || "").toLowerCase();
  return /\.(png|jpe?g|gif|webp|bmp)$/.test(name) || c.block_type === "figure" || c.block_type === "image";
}

function isAudioCitation(c: Citation): boolean {
  return (c.evidence_blocks ?? []).some(
    (b) => b.audio_start_seconds != null && b.audio_file != null,
  );
}

function VisualCitationStrip({
  citations,
  ownerId,
  onSelect,
}: {
  citations: Citation[];
  ownerId: string;
  onSelect: (c: Citation) => void;
}) {
  // Show inline thumbnails for image citations. Audio gets a special tile.
  const visual = citations.filter((c) => (isImageCitation(c) || isAudioCitation(c)) && c.doc_id);
  // Dedup by doc_id (one tile per source)
  const seen = new Set<string>();
  const tiles = visual.filter((c) => {
    if (seen.has(c.doc_id)) return false;
    seen.add(c.doc_id);
    return true;
  }).slice(0, 4);
  if (tiles.length === 0) return null;

  return (
    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">
      {tiles.map((c, i) => {
        const isAudio = isAudioCitation(c);
        const url = `${API_V1_BASE_URL}/materials/${c.doc_id}/raw?owner_id=${encodeURIComponent(ownerId)}`;
        const shortName = c.doc_name.replace(/\.[^.]+$/, "");
        const audioBlock = isAudio ? (c.evidence_blocks ?? []).find((b) => b.audio_start_seconds != null) : null;
        const startSec = audioBlock?.audio_start_seconds ?? 0;
        const mm = Math.floor(startSec / 60);
        const ss = Math.floor(startSec % 60).toString().padStart(2, "0");
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSelect(c)}
            title={c.doc_name + (isAudio ? ` (${mm}:${ss})` : "")}
            className="group relative aspect-[4/3] overflow-hidden rounded-lg border border-outline bg-surface-low transition hover:border-primary hover:shadow-md"
          >
            {isAudio ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 bg-gradient-to-br from-pink-100 to-purple-100 dark:from-pink-900/30 dark:to-purple-900/30">
                <span className="text-2xl">🎧</span>
                <span className="font-mono text-[10px] font-bold text-primary">[{mm}:{ss}]</span>
              </div>
            ) : (
              <img
                src={url}
                alt={c.doc_name}
                loading="lazy"
                className="h-full w-full object-cover transition group-hover:scale-105"
              />
            )}
            {/* Overlay with name */}
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 via-black/40 to-transparent p-1.5">
              <p className="truncate text-[10px] font-semibold text-white" title={shortName}>{shortName}</p>
              {c.page && !isAudio && (
                <p className="text-[9px] text-white/80">trang {c.page}</p>
              )}
            </div>
            {/* Citation number badge */}
            <span className="absolute top-1 left-1 rounded bg-primary/90 px-1.5 py-0.5 text-[9px] font-bold text-white">
              [{i + 1}]
            </span>
          </button>
        );
      })}
    </div>
  );
}

function CitationFooter({ citations, ownerId, onSelect }: { content: string; citations: Citation[]; ownerId: string; onSelect: (c: Citation) => void }) {
  const deduped = citations.filter((c, i, arr) => arr.findIndex((x) => x.doc_id === c.doc_id && x.page === c.page) === i).slice(0, 5);
  return (
    <div className="mt-3 border-t border-outline/40 pt-2.5">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted/70">Nguồn trích dẫn</p>
      {/* Visual tiles for images + audio */}
      <VisualCitationStrip citations={deduped} ownerId={ownerId} onSelect={onSelect} />
      <div className="mt-2 flex flex-wrap gap-1.5">
        {deduped.map((citation, i) => {
          const pageLabel = citation.page ? `trang ${citation.page}` : "";
          const tooltip = [citation.doc_name, pageLabel].filter(Boolean).join(" · ");
          return (
            <button
              key={i}
              onClick={() => onSelect(citation)}
              title={tooltip}
              className="citation-pill flex items-center gap-1 rounded-full border border-outline/60 bg-surface-low px-2.5 py-1 text-[10px] font-medium text-muted hover:border-primary/40 hover:text-primary hover:bg-primary/5"
            >
              {citationFileIcon(citation.doc_name)}
              <span className="truncate max-w-[130px]">{citation.doc_name.replace(/\.[^.]+$/, "")}</span>
              {citation.page && <span className="shrink-0 text-muted/50">p.{citation.page}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Suggestion chips ─────────────────────────────────────────────────────────

type Suggestion = { label: string; fill?: string; action?: () => void };

type SmartPrompt = {
  label: string;
  variants: Array<(target: string, primary: string) => string>;
};

const SMART_PROMPTS: SmartPrompt[] = [
  {
    label: "Tóm tắt",
    variants: [
      (target) => `Tóm tắt ${target} thành 5 ý chính dễ học.`,
      (target) => `Tóm tắt nội dung quan trọng nhất trong ${target}, ưu tiên khái niệm và kết luận.`,
      (target) => `Tạo bản tóm tắt ${target} theo cấu trúc: bối cảnh, ý chính, ví dụ, điều cần nhớ.`,
      (target) => `Tóm tắt ${target} như một ghi chú ôn thi cho sinh viên.`,
    ],
  },
  {
    label: "Ôn tập",
    variants: [
      (target) => `Tạo 10 câu hỏi ôn tập kèm đáp án dựa trên ${target}.`,
      (target) => `Tạo bộ câu hỏi trắc nghiệm và tự luận để kiểm tra mức hiểu ${target}.`,
      (target) => `Hỏi tôi từng câu một để ôn tập ${target}, bắt đầu từ mức dễ.`,
      (target) => `Tạo danh sách các câu hỏi có khả năng xuất hiện trong bài kiểm tra từ ${target}.`,
    ],
  },
  {
    label: "Giải thích",
    variants: [
      (target) => `Giải thích các khái niệm chính trong ${target} theo cách dễ hiểu.`,
      (target, primary) => `Chọn 5 khái niệm khó nhất trong "${primary}" và giải thích bằng ví dụ đơn giản.`,
      (target) => `Giải thích ${target} cho người mới bắt đầu.`,
      (target) => `Tìm các thuật ngữ quan trọng trong ${target} và giải thích ngắn gọn từng thuật ngữ.`,
    ],
  },
  {
    label: "Tìm điểm quan trọng",
    variants: [
      (target) => `Chỉ ra những phần quan trọng nhất trong ${target} mà tôi nên ưu tiên học.`,
      (target) => `Liệt kê các luận điểm, công thức hoặc định nghĩa cốt lõi cần ghi nhớ trong ${target}.`,
      (target) => `Tìm các đoạn có giá trị nhất trong ${target} để trích dẫn hoặc dùng làm bằng chứng.`,
      (target) => `Cho tôi biết phần nào trong ${target} dễ bị bỏ sót nhưng quan trọng.`,
    ],
  },
];

function cleanSourceName(name: string) {
  return name.replace(/\.[a-z0-9]+$/i, "").replace(/[_-]+/g, " ").trim() || name;
}

function formatPromptTarget(names: string[], collectionName: string) {
  const cleaned = names.map(cleanSourceName).filter(Boolean);
  if (cleaned.length === 1) return `"${cleaned[0]}"`;
  if (cleaned.length === 2) return `"${cleaned[0]}" và "${cleaned[1]}"`;
  if (cleaned.length > 2) return `các tài liệu "${cleaned[0]}", "${cleaned[1]}" và ${cleaned.length - 2} tài liệu khác`;
  return collectionName ? `bộ tài liệu "${collectionName}"` : "bộ tài liệu này";
}

function NoSourcesCallout({ onOpenSources }: { onOpenSources: () => void }) {
  return (
    <div className="mx-auto mt-8 max-w-md rounded-lg border border-amber-200 bg-amber-50 px-4 py-5 text-center shadow-sm">
      <p className="text-sm font-semibold text-text">Thêm tài liệu để bắt đầu hỏi đáp có căn cứ</p>
      <p className="mt-1 text-xs leading-5 text-muted">
        Noelys cần nguồn như PDF, DOCX, ảnh hoặc bảng dữ liệu để trả lời kèm trích dẫn.
      </p>
      <button
        type="button"
        onClick={onOpenSources}
        className="mt-3 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white transition hover:bg-primary/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
      >
        Mở Sources
      </button>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

type ChatPanelProps = {
  onOpenSources: () => void;
  onOpenEvidence: () => void;
  onTraceGraph?: () => void;
  onTabChange?: (tab: StudioTab) => void;
};

export default function ChatPanel({ onOpenSources, onOpenEvidence, onTraceGraph, onTabChange }: ChatPanelProps) {
  const {
    workspace,
    scopedMaterialIds,
    sourceScopeMode,
    setSelectedCitation,
    setActiveCitations,
    setActiveQueryContext,
    chatDraft,
    setChatDraft,
    readySourceCount,
    readySources,
  } = useWorkspace();
  const currentChatStorageKey = chatStorageKey(workspace.ownerId, workspace.collectionId);
  const suppressNextPersistRef = useRef(false);

  const makeIntroMessage = (): ChatMessage => ({
    id: "intro",
    role: "assistant",
    content: getIntroMessage(workspace.language),
  });

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const stored = localStorage.getItem(currentChatStorageKey);
      return stored ? JSON.parse(stored) : [makeIntroMessage()];
    } catch {
      return [makeIntroMessage()];
    }
  });

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<string | null>(null);
  const [promptVersion, setPromptVersion] = useState(0);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [attachedImage, setAttachedImage] = useState<File | null>(null);
  const [attachedImagePreview, setAttachedImagePreview] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const confirmClearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const hasScope = Boolean(workspace.collectionId ? sourceScopeMode === "all" || scopedMaterialIds.length : scopedMaterialIds.length);
  const activeSourceNames = useMemo(() => {
    const selected = sourceScopeMode === "selected" && scopedMaterialIds.length
      ? readySources.filter((source) => scopedMaterialIds.includes(source.materialId))
      : readySources;
    return selected.map((source) => source.name);
  }, [readySources, scopedMaterialIds, sourceScopeMode]);
  const visibleSmartPrompts = useMemo(() => {
    const target = formatPromptTarget(activeSourceNames, workspace.collectionName);
    const primary = cleanSourceName(activeSourceNames[0] ?? workspace.collectionName ?? "tài liệu này");
    return SMART_PROMPTS.map((prompt, index) => ({
      label: prompt.label,
      fill: prompt.variants[(promptVersion + index) % prompt.variants.length](target, primary),
    }));
  }, [activeSourceNames, promptVersion, workspace.collectionName]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    suppressNextPersistRef.current = true;
    try {
      const stored = localStorage.getItem(currentChatStorageKey);
      setMessages(stored ? JSON.parse(stored) : [makeIntroMessage()]);
    } catch {
      setMessages([makeIntroMessage()]);
    }
  }, [currentChatStorageKey, workspace.language]);

  useEffect(() => {
    if (!chatDraft) {
      return;
    }
    setQuestion(chatDraft);
    setChatDraft(null);
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }, [chatDraft, setChatDraft]);

  useEffect(() => {
    if (suppressNextPersistRef.current) {
      suppressNextPersistRef.current = false;
      return;
    }
    try { localStorage.setItem(currentChatStorageKey, JSON.stringify(messages)); } catch { /* quota */ }
  }, [currentChatStorageKey, messages]);

  function clearChat() {
    setMessages([makeIntroMessage()]);
    setError(null);
    setActiveQueryContext(null);
  }

  function requestClearChat() {
    setConfirmingClear(true);
    if (confirmClearTimerRef.current) clearTimeout(confirmClearTimerRef.current);
    confirmClearTimerRef.current = setTimeout(() => setConfirmingClear(false), 4000);
  }

  function confirmClearChat() {
    if (confirmClearTimerRef.current) clearTimeout(confirmClearTimerRef.current);
    setConfirmingClear(false);
    clearChat();
  }

  function cancelClearChat() {
    if (confirmClearTimerRef.current) clearTimeout(confirmClearTimerRef.current);
    setConfirmingClear(false);
  }

  function selectCitation(citation: Citation) {
    setSelectedCitation(citation);
    onOpenEvidence();
  }

  function onPickImage(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!/^image\/(png|jpe?g|webp|gif|bmp)$/i.test(file.type)) {
      setError("Chỉ hỗ trợ ảnh PNG, JPG, WEBP, GIF, BMP.");
      e.target.value = "";
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError("Ảnh vượt quá 8 MB.");
      e.target.value = "";
      return;
    }
    setError(null);
    setAttachedImage(file);
    if (attachedImagePreview) URL.revokeObjectURL(attachedImagePreview);
    setAttachedImagePreview(URL.createObjectURL(file));
    e.target.value = "";
  }

  function clearAttachedImage() {
    if (attachedImagePreview) URL.revokeObjectURL(attachedImagePreview);
    setAttachedImage(null);
    setAttachedImagePreview(null);
  }

  useEffect(() => () => {
    if (attachedImagePreview) URL.revokeObjectURL(attachedImagePreview);
  }, [attachedImagePreview]);

  async function handleImageQuery(image: File, caption: string) {
    if (!hasScope) {
      setError(null);
      setQuestion("");
      clearAttachedImage();
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: caption || `[Ảnh ${image.name}]` },
        { id: crypto.randomUUID(), role: "assistant", content: "Hãy thêm tài liệu ở panel bên trái để tìm bằng ảnh có căn cứ." },
      ]);
      return;
    }

    setError(null);
    setAgentStatus("Đang tìm hình ảnh tương tự trong tài liệu...");
    setLoading(true);
    const userContent = caption ? `${caption}\n\n📎 ${image.name}` : `📎 ${image.name}`;
    setQuestion("");
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: userContent }]);
    const assistantId = crypto.randomUUID();

    try {
      const response = await askQuestionWithImage({
        ownerId: workspace.ownerId,
        collectionId: workspace.collectionId || null,
        materialIds: scopedMaterialIds,
        conversationId: currentChatStorageKey,
        queryText: caption || undefined,
        topK: workspace.topK ?? undefined,
        answerLanguage: workspace.language,
        image,
      });
      const content = response.was_refused ? friendlyRefusal(response.refusal_reason) : response.answer;
      setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content, response }]);
      if (response.citations.length > 0) {
        setActiveCitations(response.citations);
        selectCitation(response.citations[0]);
      }
      setActiveQueryContext({ question: caption || `[Ảnh ${image.name}]`, response, createdAt: new Date().toISOString() });
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
      setAgentStatus(null);
      clearAttachedImage();
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (loading || streamingId) return;

    if (attachedImage) {
      await handleImageQuery(attachedImage, trimmed);
      return;
    }

    if (!trimmed) return;

    const localReply = casualReply(trimmed, hasScope);
    if (localReply) {
      setError(null);
      setQuestion("");
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: trimmed },
        { id: crypto.randomUUID(), role: "assistant", content: localReply },
      ]);
      return;
    }

    if (!hasScope) {
      setError(null);
      setQuestion("");
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: trimmed },
        { id: crypto.randomUUID(), role: "assistant", content: "Hãy thêm tài liệu ở panel bên trái để bắt đầu hỏi đáp có căn cứ." },
      ]);
      return;
    }

    setError(null);
    setAgentStatus("Đang chuẩn bị tìm bằng chứng...");
    setLoading(true);
    setQuestion("");
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);

    const assistantId = crypto.randomUUID();

    await askQuestionStream(
      {
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        material_ids: scopedMaterialIds,
        conversation_id: currentChatStorageKey,
        query: trimmed,
        top_k: workspace.topK,
        answer_language: workspace.language,
      },
      {
        onToken(token) {
          setLoading(false);
          setStreamingId(assistantId);
          setMessages((prev) => {
            const exists = prev.some((m) => m.id === assistantId);
            if (exists) {
              return prev.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + token } : m,
              );
            }
            return [...prev, { id: assistantId, role: "assistant", content: token }];
          });
        },
        onDone(response) {
          setStreamingId(null);
          setLoading(false);
          setAgentStatus(null);
          const content = response.was_refused
            ? friendlyRefusal(response.refusal_reason)
            : response.answer;
          setMessages((prev) => {
            const exists = prev.some((m) => m.id === assistantId);
            if (exists) {
              return prev.map((m) =>
                m.id === assistantId ? { ...m, content, response } : m,
              );
            }
            return [...prev, { id: assistantId, role: "assistant", content, response }];
          });
          if (response.citations.length > 0) {
            setActiveCitations(response.citations);
            selectCitation(response.citations[0]);
          }
          setActiveQueryContext({ question: trimmed, response, createdAt: new Date().toISOString() });
        },
        onError(message) {
          setStreamingId(null);
          setLoading(false);
          setAgentStatus(null);
          setMessages((prev) => prev.filter((m) => m.id !== assistantId));
          setError(friendlyError(new Error(message)));
        },
        onAgentStep(step) {
          setLoading(true);
          setAgentStatus(agentStepLabel(step.name));
        },
      },
    );
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-[48px] shrink-0 items-center justify-between bg-white/80 px-5 z-10 section-divider" style={{ backdropFilter: 'blur(8px)' }}>
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            aria-label="Mở danh sách nguồn"
            className="lg:hidden rounded-lg p-1.5 text-muted hover:bg-slate-100 hover:text-primary transition"
            onClick={onOpenSources}
          >
            <Library size={17} />
          </button>
          {workspace.collectionName ? (
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-[14px] text-text truncate max-w-[200px]">{workspace.collectionName}</span>
              <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full border ${hasScope ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"}`}>
                {hasScope
                  ? sourceScopeMode === "selected"
                    ? `${scopedMaterialIds.length} nguồn`
                    : "Tất cả nguồn"
                  : "Chưa có nguồn"}
              </span>
            </div>
          ) : (
            <span className="font-heading font-semibold text-[13px] text-muted">Chưa chọn bộ tài liệu</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold text-muted/60 uppercase tracking-wider">{workspace.language}</span>
          <div className="h-4 w-px bg-outline/40" />
          {confirmingClear ? (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-red-600">Xóa toàn bộ?</span>
              <button
                type="button"
                onClick={confirmClearChat}
                aria-label="Xác nhận xóa chat"
                className="rounded-md px-2 py-0.5 text-[11px] font-bold text-red-600 hover:bg-red-50 transition"
              >
                Xóa
              </button>
              <button
                type="button"
                onClick={cancelClearChat}
                aria-label="Hủy xóa chat"
                className="rounded-md px-2 py-0.5 text-[11px] font-semibold text-muted hover:bg-slate-100 transition"
              >
                Hủy
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={requestClearChat}
              title="Xóa nội dung chat"
              aria-label="Xóa nội dung chat"
              className="flex h-7 w-7 items-center justify-center rounded-lg text-muted/60 hover:bg-red-50 hover:text-red-500 transition"
            >
              <Trash2 size={13} />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mx-5 mt-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Messages */}
      <div role="log" aria-live="polite" aria-label="Chat messages" className="flex-1 overflow-y-auto p-4 md:p-6 space-y-5" style={{ background: 'linear-gradient(180deg, var(--c-surface-low) 0%, var(--c-surface-mid) 100%)' }}>
        {!hasScope && messages.length <= 1 && <NoSourcesCallout onOpenSources={onOpenSources} />}
        {messages.map((message) => {
          const isStreaming = message.id === streamingId;
          return (
            <article key={message.id} className={`flex chat-message-animate ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[88%] md:max-w-[78%] px-5 py-4 ${message.role === "user" ? "rounded-2xl rounded-br-md text-white shadow-md" : "rounded-2xl rounded-bl-md bg-white border border-outline/30 shadow-sm"}`}
                style={message.role === "user" ? { background: 'linear-gradient(135deg, #006591 0%, #0284c7 100%)' } : undefined}
              >
                {message.response && !message.response.was_refused ? (
                  <MessageContent
                    content={message.content}
                    citations={message.response.citations}
                    onCitationClick={(idx) => selectCitation(message.response!.citations[idx])}
                  />
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">
                    {message.content}
                    {isStreaming && (
                      <span className="ml-0.5 inline-block w-0.5 h-4 bg-current align-middle animate-pulse" />
                    )}
                  </p>
                )}

                {/* Compact meta strip + collapsible details */}
                {message.response && !message.response.was_refused && (
                  <AnswerMeta response={message.response} onTraceGraph={onTraceGraph} />
                )}

                {/* Citations footer */}
                {message.response && !message.response.was_refused && message.response.citations.length > 0 && (
                  <CitationFooter
                    content={message.content}
                    citations={message.response.citations}
                    ownerId={workspace.ownerId}
                    onSelect={selectCitation}
                  />
                )}
              </div>
            </article>
          );
        })}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-outline shadow-sm rounded-2xl px-5 py-4 flex items-center gap-3 text-sm text-muted">
              <Loader2 className="animate-spin text-primary" size={18} />
              {agentStatus ?? "Đang tìm kiếm và tổng hợp..."}
            </div>
          </div>
        )}
        <div ref={bottomRef} className="h-4" />
      </div>

      {/* Input */}
      <div className="shrink-0 bg-white/90 border-t border-outline/30 p-4 md:p-5" style={{ backdropFilter: 'blur(8px)' }}>
        <div className="mx-auto max-w-4xl">
          {hasScope && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted/50">
                Gợi ý hỏi
              </span>
              {visibleSmartPrompts.map((s) => (
                <button
                  key={s.label}
                  type="button"
                  className="suggestion-chip shrink-0 rounded-full border border-outline/50 bg-white px-3 py-1 text-[11px] font-medium text-muted hover:border-primary/40 hover:text-primary hover:bg-primary/5"
                  onClick={() => {
                    if (!s.fill) return;
                    setQuestion(s.fill);
                    setPromptVersion((value) => value + 1);
                    setError(null);
                    requestAnimationFrame(() => {
                      const el = textareaRef.current;
                      if (el) { el.focus(); el.setSelectionRange(s.fill!.length, s.fill!.length); }
                    });
                  }}
                >
                  {s.label}
                </button>
              ))}
              <span className="text-[9px] text-muted/50 font-medium">
                {sourceScopeMode === "selected" ? `${readySourceCount} nguồn đã chọn` : "dùng bộ tài liệu hiện tại"}
              </span>
            </div>
          )}
          <form
            className="chat-input-wrapper rounded-xl border border-outline/40 bg-white shadow-sm overflow-hidden"
            onSubmit={handleSubmit}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/gif,image/bmp"
              className="hidden"
              onChange={onPickImage}
            />
            {attachedImagePreview && (
              <div className="flex items-center gap-2 border-b border-outline/30 bg-surface-low px-3 py-2">
                <img
                  src={attachedImagePreview}
                  alt={attachedImage?.name ?? "preview"}
                  className="h-12 w-12 rounded-md object-cover ring-1 ring-outline/40"
                />
                <div className="flex-1 min-w-0">
                  <p className="truncate text-xs font-semibold text-text">{attachedImage?.name}</p>
                  <p className="text-[10px] text-muted">
                    Tìm bằng hình ảnh{question.trim() ? " · có chú thích" : " · không chú thích"}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={clearAttachedImage}
                  className="rounded-md p-1 text-muted hover:bg-surface hover:text-text"
                  title="Bỏ ảnh"
                >
                  <X size={14} />
                </button>
              </div>
            )}
            <div className="flex items-end gap-1 pl-1 pr-1.5 py-1">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading || !!streamingId}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted/70 transition hover:bg-primary/8 hover:text-primary disabled:opacity-40"
                title="Đính kèm ảnh để tìm theo hình"
              >
                <ImagePlus size={17} strokeWidth={1.75} />
              </button>
              <textarea
                ref={textareaRef}
                className="flex-1 resize-none self-center border-0 bg-transparent px-1 py-2 text-sm leading-5 outline-none ring-0 focus:border-0 focus:outline-none focus:ring-0 min-h-[36px] placeholder:text-muted/40"
                style={{ maxHeight: "8rem" }}
                placeholder={
                  attachedImage
                    ? "Mô tả ảnh hoặc bỏ trống để chỉ tìm bằng ảnh..."
                    : hasScope
                      ? "Hỏi bất cứ điều gì về tài liệu của bạn..."
                      : "Hỏi thông thường hoặc thêm tài liệu để hỏi đáp có căn cứ..."
                }
                value={question}
                rows={1}
                onChange={(e) => {
                  setQuestion(e.target.value);
                  const el = e.target;
                  el.style.height = "auto";
                  el.style.height = Math.min(el.scrollHeight, 128) + "px";
                }}
                onKeyDown={handleKeyDown}
              />
              <button
                disabled={loading || !!streamingId || (!question.trim() && !attachedImage)}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-white transition disabled:cursor-not-allowed disabled:opacity-30"
                style={{ background: (!question.trim() && !attachedImage) || loading || !!streamingId ? '#94a3b8' : 'linear-gradient(135deg, #006591 0%, #0ea5e9 100%)' }}
                title="Gửi"
              >
                {loading || streamingId ? <Loader2 className="animate-spin" size={15} /> : <Send size={15} strokeWidth={2} className="-ml-px" />}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
