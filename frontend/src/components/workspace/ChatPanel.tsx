import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { AlertCircle, BookOpen, GitBranch, Loader2, Network, Send, Share2, Sparkles, Trash2, Library } from "lucide-react";
import { Citation, QueryResponse, askQuestion } from "../../api/client";
import { useWorkspace } from "../../state/workspace";
import { StudioTab } from "../../pages/WorkspacePage";
import MarkdownRenderer from "../MarkdownRenderer";

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
  if (language === "vi") return "Xin chào! Tôi là trợ lý Prism. Hãy tải tài liệu lên bên trái để bắt đầu hỏi đáp có căn cứ, hoặc hỏi tôi bất cứ điều gì.";
  if (language === "zh") return "你好！我是 Prism 助手。请在左侧上传文档以开始有依据的问答。";
  if (language === "ja") return "こんにちは！Prismアシスタントです。左側にドキュメントをアップロードして始めてください。";
  if (language === "ko") return "안녕하세요! Prism 어시스턴트입니다. 왼쪽에 문서를 업로드하여 시작하세요.";
  return "Welcome to Prism! Upload some sources on the left to start grounded Q&A, or ask me anything.";
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
    return "Tôi là trợ lý AI của Prism, chuyên về Document Intelligence và Graph RAG.";
  }
  return null;
}

// ─── Confidence badge ─────────────────────────────────────────────────────────

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 70 ? "bg-emerald-100 text-emerald-700" : pct >= 40 ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700";
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold ${cls}`}>{pct}%</span>;
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

// ─── Citation footer ──────────────────────────────────────────────────────────

function CitationFooter({ citations, onSelect }: { content: string; citations: Citation[]; onSelect: (c: Citation) => void }) {
  const deduped = citations.filter((c, i, arr) => arr.findIndex((x) => x.doc_id === c.doc_id && x.page === c.page) === i).slice(0, 5);
  return (
    <div className="mt-3 border-t border-outline/50 pt-2.5">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted">Sources</p>
      <div className="flex flex-wrap gap-1.5">
        {deduped.map((citation, i) => {
          const pageLabel = citation.page ? ` p.${citation.page}` : "";
          return (
            <button
              key={i}
              onClick={() => onSelect(citation)}
              className="flex items-center gap-1 rounded-full border border-outline bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-muted hover:border-primary/40 hover:text-primary transition"
            >
              <span className="text-[11px] text-muted truncate max-w-[140px]">
                {citation.doc_name}
                {pageLabel && <span className="ml-1 text-muted/60">{pageLabel}</span>}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Suggestion chips ─────────────────────────────────────────────────────────

type Suggestion = { label: string; fill?: string; action?: () => void };

function NoSourcesCallout({ onOpenSources }: { onOpenSources: () => void }) {
  return (
    <div className="mx-auto mt-8 max-w-md rounded-lg border border-amber-200 bg-amber-50 px-4 py-5 text-center shadow-sm">
      <p className="text-sm font-semibold text-text">Thêm tài liệu để bắt đầu hỏi đáp có căn cứ</p>
      <p className="mt-1 text-xs leading-5 text-muted">
        Prism cần nguồn như PDF, DOCX, ảnh hoặc bảng dữ liệu để trả lời kèm trích dẫn.
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
  onTabChange?: (tab: StudioTab) => void;
};

export default function ChatPanel({ onOpenSources, onOpenEvidence, onTabChange }: ChatPanelProps) {
  const { workspace, scopedMaterialIds, setSelectedCitation, setActiveCitations } = useWorkspace();
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
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const hasScope = Boolean(workspace.collectionId || scopedMaterialIds.length);

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
    if (suppressNextPersistRef.current) {
      suppressNextPersistRef.current = false;
      return;
    }
    try { localStorage.setItem(currentChatStorageKey, JSON.stringify(messages)); } catch { /* quota */ }
  }, [currentChatStorageKey, messages]);

  function clearChat() {
    setMessages([makeIntroMessage()]);
    setError(null);
  }

  function selectCitation(citation: Citation) {
    setSelectedCitation(citation);
    onOpenEvidence();
  }

  const suggestions: Suggestion[] = [
    {
      label: "Tóm tắt",
      action: () => onTabChange?.("studio"),
    },
    {
      label: "So sánh nguồn",
      action: () => onTabChange?.("compare"),
    },
    {
      label: "Study Guide",
      action: () => onTabChange?.("studio"),
    },
    {
      label: "Mindmap",
      action: () => onTabChange?.("mindmap"),
    },
    {
      label: "Graph",
      action: () => onTabChange?.("graph"),
    },
  ];

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

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
    setLoading(true);
    setQuestion("");
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", content: trimmed }]);

    try {
      const response = await askQuestion({
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        material_ids: workspace.collectionId ? [] : scopedMaterialIds,
        conversation_id: currentChatStorageKey,
        query: trimmed,
        top_k: workspace.topK,
        answer_language: workspace.language,
      });
      if (response.citations.length > 0) {
        setActiveCitations(response.citations);
        selectCitation(response.citations[0]);
      }
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.was_refused ? friendlyRefusal(response.refusal_reason) : response.answer,
          response,
        },
      ]);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
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
      <div className="flex shrink-0 items-center justify-between border-b border-outline bg-white px-6 py-3 shadow-sm z-10">
        <div className="flex items-center gap-3">
          <button
            type="button"
            aria-label="Mở danh sách nguồn"
            className="lg:hidden rounded-md p-2 text-muted hover:bg-slate-100 hover:text-primary"
            onClick={onOpenSources}
          >
            <Library size={18} />
          </button>
          {workspace.collectionName ? (
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-text truncate max-w-[200px]">{workspace.collectionName}</span>
              <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${hasScope ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                {hasScope ? "Grounded" : "No sources"}
              </span>
            </div>
          ) : (
            <span className="font-heading font-semibold text-muted">No active collection</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-semibold text-muted uppercase tracking-wider">{workspace.language}</span>
          <div className="h-4 w-px bg-outline" />
          <button
            type="button"
            onClick={clearChat}
            title="Clear chat"
            aria-label="Xóa nội dung chat"
            className="flex h-7 w-7 items-center justify-center rounded-md border border-outline text-muted hover:border-red-300 hover:bg-red-50 hover:text-red-500 transition"
          >
            <Trash2 size={13} />
          </button>
        </div>
      </div>

      {error && (
        <div className="mx-5 mt-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {!hasScope && messages.length <= 1 && <NoSourcesCallout onOpenSources={onOpenSources} />}
        {messages.map((message) => (
          <article key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[90%] md:max-w-[80%] rounded-2xl px-5 py-4 ${message.role === "user" ? "bg-primary text-white" : "bg-white border border-outline shadow-sm"}`}>
              {message.response && !message.response.was_refused ? (
                <MessageContent
                  content={message.content}
                  citations={message.response.citations}
                  onCitationClick={(idx) => selectCitation(message.response!.citations[idx])}
                />
              ) : (
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
              )}

              {/* Confidence badge */}
              {message.response && !message.response.was_refused && (
                <div className="mt-2 flex items-center gap-2">
                  <ConfidenceBadge value={message.response.confidence} />
                </div>
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
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-outline shadow-sm rounded-2xl px-5 py-4 flex items-center gap-3 text-sm text-muted">
              <Loader2 className="animate-spin text-primary" size={18} />
              Đang tìm kiếm và tổng hợp...
            </div>
          </div>
        )}
        <div ref={bottomRef} className="h-4" />
      </div>

      {/* Input */}
      <div className="shrink-0 bg-white border-t border-outline p-4 md:p-6">
        <div className="mx-auto max-w-4xl">
          <form
            className="relative rounded-xl border border-outline bg-white shadow-sm focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all overflow-hidden"
            onSubmit={handleSubmit}
          >
            <textarea
              ref={textareaRef}
              className="w-full resize-none bg-transparent px-4 py-4 pr-14 text-sm outline-none max-h-32 min-h-[60px]"
              placeholder={hasScope ? "Hỏi bất cứ điều gì về tài liệu của bạn..." : "Hỏi thông thường hoặc thêm tài liệu để hỏi đáp có căn cứ..."}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={Math.min(5, question.split("\n").length || 1)}
            />
            <div className="absolute right-2 bottom-2">
              <button
                disabled={loading || !question.trim()}
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white disabled:opacity-50 transition hover:bg-primary/90"
              >
                {loading ? <Loader2 className="animate-spin" size={14} /> : <Send size={14} className="ml-0.5" />}
              </button>
            </div>
          </form>

          {/* Suggestion chips */}
          <div className="flex flex-wrap items-center gap-2 mt-3 overflow-x-auto pb-1 no-scrollbar">
            {suggestions.map((s) => (
              <button
                key={s.label}
                type="button"
                className="shrink-0 rounded-full border border-outline bg-white px-3 py-1 text-[11px] font-medium text-muted hover:text-primary hover:border-primary/30 transition whitespace-nowrap"
                onClick={() => {
                  if (s.action) {
                    s.action();
                  } else if (s.fill) {
                    setQuestion(s.fill);
                    setError(null);
                    requestAnimationFrame(() => {
                      const el = textareaRef.current;
                      if (el) { el.focus(); el.setSelectionRange(s.fill!.length, s.fill!.length); }
                    });
                  }
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
