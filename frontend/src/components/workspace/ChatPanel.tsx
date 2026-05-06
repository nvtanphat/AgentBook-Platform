import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, FileText, Image, Loader2, Network, Send, Table2, Trash2, Library } from "lucide-react";
import { Citation, QueryResponse, askQuestionStream } from "../../api/client";
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
    trace_graph: "Đang truy vết quan hệ trên Knowledge Graph...",
    verify_coverage: "Đang kiểm tra độ phủ nguồn...",
    repair_retrieval: "Đang bổ sung bằng chứng còn thiếu...",
    rerank_evidence: "Đang xếp hạng bằng chứng phù hợp...",
    synthesize_answer: "Đang tổng hợp câu trả lời...",
    verify_claims: "Đang kiểm chứng câu trả lời...",
  };
  return labels[name] ?? `Đang xử lý: ${name.replace(/_/g, " ")}`;
}

function agentTraceStepLabel(name: string): string {
  return agentStepLabel(name).replace(/^Đang\s+/i, "").replace(/\.\.\.$/, "");
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

function MessageContent({ content, citations, response, onCitationClick }: {
  content: string;
  citations: Citation[];
  response?: QueryResponse;
  onCitationClick: (idx: number) => void;
}) {
  return (
    <>
      <MarkdownRenderer
        text={content}
        onCitationClick={(ref) => {
          if (ref >= 0 && ref < citations.length) onCitationClick(ref);
        }}
      />

      {/* Reasoning trace */}
      {response?.reasoning_path && response.reasoning_path.length > 0 && (
        <ReasoningTrace
          steps={response.reasoning_path}
          onStepHover={() => undefined}
        />
      )}
    </>
  );
}

function AgentBadges({ response }: { response: QueryResponse }) {
  const coverage = response.coverage;
  const trace = response.agent_trace;
  const verification = trace?.verification;
  const completeCoverage = coverage ? coverage.covered_count >= coverage.requested_count : true;
  const verified = verification?.verdict === "supported";
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
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
      {trace?.repair_attempted && (
        <span className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-bold text-blue-700">
          Đã bổ sung truy xuất
        </span>
      )}
      {trace && (
        <span className="inline-flex items-center rounded-full border border-outline bg-slate-50 px-2 py-0.5 text-[10px] font-bold text-muted">
          Kế hoạch: {trace.plan_type.replace(/_/g, " ")}
        </span>
      )}
    </div>
  );
}

function AgentTracePanel({ response }: { response: QueryResponse }) {
  const [open, setOpen] = useState(false);
  const trace = response.agent_trace;
  if (!trace) return null;
  return (
    <div className="mt-2 rounded-lg border border-outline/50 bg-slate-50/80">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
      >
        <span className="text-[11px] font-bold uppercase tracking-wider text-muted">Luồng xử lý agent</span>
        {open ? <ChevronUp size={13} className="text-muted" /> : <ChevronDown size={13} className="text-muted" />}
      </button>
      {open && (
        <div className="border-t border-outline/50 px-3 py-2">
          <div className="mb-2 flex flex-wrap gap-1.5 text-[10px] font-semibold text-muted">
            <span className="rounded border border-outline bg-white px-2 py-0.5">{trace.plan_type}</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">{trace.steps.length} steps</span>
            {trace.repair_attempted && <span className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-blue-700">repair used</span>}
          </div>
          <div className="space-y-1.5">
            {trace.steps.map((step, index) => (
              <div key={`${step.name}-${index}`} className="rounded border border-outline bg-white px-2 py-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-text">{index + 1}. {agentTraceStepLabel(step.name)}</span>
                  <span className={`rounded-full px-1.5 py-0.5 text-[9px] font-bold ${
                    step.status === "completed" ? "bg-emerald-50 text-emerald-700" :
                    step.status === "failed" ? "bg-red-50 text-red-700" :
                    "bg-slate-100 text-muted"
                  }`}>
                    {step.status}
                  </span>
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5 text-[10px] text-muted">
                  {step.sources_requested != null && <span>{step.sources_covered ?? 0}/{step.sources_requested} nguồn</span>}
                  {step.evidence_count != null && <span>{step.evidence_count} bằng chứng</span>}
                  {step.warning && <span className="text-amber-700">{step.warning}</span>}
                </div>
              </div>
            ))}
          </div>
          {trace.verification && (
            <div className="mt-2 rounded border border-outline bg-white px-2 py-1.5 text-[10px] text-muted">
              Kiểm chứng: <span className="font-bold text-text">{trace.verification.verdict}</span>
              {" "}({Math.round(trace.verification.confidence * 100)}%)
              {trace.verification.warning && <span className="ml-1 text-amber-700">{trace.verification.warning}</span>}
            </div>
          )}
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

function CitationFooter({ citations, onSelect }: { content: string; citations: Citation[]; onSelect: (c: Citation) => void }) {
  const deduped = citations.filter((c, i, arr) => arr.findIndex((x) => x.doc_id === c.doc_id && x.page === c.page) === i).slice(0, 5);
  return (
    <div className="mt-3 border-t border-outline/40 pt-2.5">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted/70">Nguồn trích dẫn</p>
      <div className="flex flex-wrap gap-1.5">
        {deduped.map((citation, i) => {
          const pageLabel = citation.page ? `trang ${citation.page}` : "";
          const tooltip = [citation.doc_name, pageLabel].filter(Boolean).join(" · ");
          return (
            <button
              key={i}
              onClick={() => onSelect(citation)}
              title={tooltip}
              className="citation-pill flex items-center gap-1 rounded-full border border-outline/60 bg-slate-50/80 px-2.5 py-1 text-[10px] font-medium text-muted hover:border-primary/40 hover:text-primary hover:bg-primary/5"
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
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
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

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading || streamingId) return;

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
    setAgentStatus("Đang lập kế hoạch agentic RAG...");
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
      <div role="log" aria-live="polite" aria-label="Chat messages" className="flex-1 overflow-y-auto p-4 md:p-6 space-y-5" style={{ background: 'linear-gradient(180deg, #fafbff 0%, #f5f7fc 100%)' }}>
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
                    response={message.response}
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

                {/* Confidence badge */}
                {message.response && !message.response.was_refused && (
                  <div className="mt-2 flex flex-col gap-2">
                    <ConfidenceBadge value={message.response.confidence} />
                    <AgentBadges response={message.response} />
                    <AgentTracePanel response={message.response} />
                  </div>
                )}

                {message.response && !message.response.was_refused && message.response.citations.length > 0 && (
                  <TraceGraphAction onTraceGraph={onTraceGraph} />
                )}

                {/* Citations footer */}
                {message.response && !message.response.was_refused && message.response.citations.length > 0 && (
                  <CitationFooter
                    content={message.content}
                    citations={message.response.citations}
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
            className="chat-input-wrapper relative rounded-xl border border-outline/40 bg-white shadow-sm overflow-hidden"
            onSubmit={handleSubmit}
          >
            <textarea
              ref={textareaRef}
              className="w-full resize-none bg-transparent px-4 py-3.5 pr-14 text-sm outline-none min-h-[52px] placeholder:text-muted/40"
              style={{ maxHeight: "8rem" }}
              placeholder={hasScope ? "Hỏi bất cứ điều gì về tài liệu của bạn..." : "Hỏi thông thường hoặc thêm tài liệu để hỏi đáp có căn cứ..."}
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
            <div className="absolute right-2 bottom-2">
              <button
                disabled={loading || !!streamingId || !question.trim()}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-white disabled:opacity-40 transition shadow-sm hover:shadow-md"
                style={{ background: loading || !!streamingId || !question.trim() ? '#94a3b8' : 'linear-gradient(135deg, #006591 0%, #0ea5e9 100%)' }}
              >
                {loading || streamingId ? <Loader2 className="animate-spin" size={14} /> : <Send size={14} className="ml-0.5" />}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
