import React from "react";
import { API_BASE_URL } from "../api/client";

function resolveImageUrl(url: string): string {
  // Backend may emit relative /api/v1/... URLs for inline figure citations so the
  // markdown isn't tied to a specific deployment host. Resolve against API_BASE_URL.
  if (url.startsWith("/api/")) return `${API_BASE_URL}${url}`;
  return url;
}

// ──────────────────────────────────────────────────────────────
// Shared markdown table parsing (used by both block renderer and SnippetRenderer)
// ──────────────────────────────────────────────────────────────

export type ParsedTable = { headers: string[]; rows: string[][] };

export function splitMarkdownRow(line: string): string[] {
  return line
    .replace(/\\\|/g, "\x00")
    .split("|")
    .slice(1, -1)
    .map((cell) => cell.replace(/\x00/g, "|").trim());
}

export function parseMarkdownTable(text: string): ParsedTable | null {
  const lines = text.trim().split("\n").filter((l) => l.trim());
  if (lines.length < 2) return null;
  if (!lines[0].trim().startsWith("|")) return null;
  if (!/^\|[\s|:*-]+\|/.test(lines[1].trim())) return null;
  return { headers: splitMarkdownRow(lines[0]), rows: lines.slice(2).map(splitMarkdownRow) };
}

type TableViewProps = { table: ParsedTable; maxRows?: number; className?: string };

export function TableView({ table, maxRows, className = "" }: TableViewProps) {
  const visible = maxRows !== undefined ? table.rows.slice(0, maxRows) : table.rows;
  const truncated = maxRows !== undefined && table.rows.length > maxRows;
  return (
    <div className={`overflow-x-auto rounded border border-outline ${className}`}>
      <table className="min-w-full text-left text-xs">
        <thead className="bg-slate-100">
          <tr>
            {table.headers.map((h, i) => (
              <th key={i} className="whitespace-nowrap border-r border-outline px-3 py-2 font-semibold text-text last:border-r-0">
                {h || <span className="italic text-muted">Col {i + 1}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {visible.map((row, ri) => (
            <tr key={ri} className="hover:bg-slate-50">
              {table.headers.map((_, ci) => (
                <td key={ci} className="max-w-[200px] truncate border-r border-slate-100 px-3 py-1.5 text-muted last:border-r-0" title={row[ci] ?? ""}>
                  {row[ci] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {truncated && (
        <div className="border-t border-outline bg-slate-50 px-3 py-1.5 text-[11px] text-muted">
          +{table.rows.length - maxRows!} more rows
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────
// Inline token types
// ──────────────────────────────────────────────────────────────

type InlineToken =
  | { t: "text"; v: string }
  | { t: "bold"; v: string }
  | { t: "italic"; v: string }
  | { t: "bold_italic"; v: string }
  | { t: "code"; v: string }
  | { t: "math"; v: string }
  | { t: "cite"; refs: number[] };

// ──────────────────────────────────────────────────────────────
// Inline parser — character-by-character with greedy matching
// ──────────────────────────────────────────────────────────────

export function parseInline(text: string, withCitations = false): InlineToken[] {
  const tokens: InlineToken[] = [];
  let i = 0;
  let buf = "";

  const flush = () => { if (buf) { tokens.push({ t: "text", v: buf }); buf = ""; } };

  while (i < text.length) {
    const ch = text[i];

    // Bold-italic: ***text***
    if (ch === "*" && text[i + 1] === "*" && text[i + 2] === "*") {
      const end = text.indexOf("***", i + 3);
      if (end !== -1) { flush(); tokens.push({ t: "bold_italic", v: text.slice(i + 3, end) }); i = end + 3; continue; }
    }

    // Bold: **text**
    if (ch === "*" && text[i + 1] === "*") {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) { flush(); tokens.push({ t: "bold", v: text.slice(i + 2, end) }); i = end + 2; continue; }
    }

    // Italic: *text* — only match if preceded by space/start and followed by non-*
    if (ch === "*" && text[i + 1] !== "*" && text[i + 1] !== " ") {
      const end = text.indexOf("*", i + 1);
      if (end !== -1 && text[end + 1] !== "*") { flush(); tokens.push({ t: "italic", v: text.slice(i + 1, end) }); i = end + 1; continue; }
    }

    // Inline code: `text`
    if (ch === "`" && text[i + 1] !== "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) { flush(); tokens.push({ t: "code", v: text.slice(i + 1, end) }); i = end + 1; continue; }
    }

    // Inline math: $text$ (not $$)
    if (ch === "$" && text[i + 1] !== "$" && text[i + 1] !== " ") {
      const end = text.indexOf("$", i + 1);
      if (end !== -1 && text[end + 1] !== "$") { flush(); tokens.push({ t: "math", v: text.slice(i + 1, end) }); i = end + 1; continue; }
    }

    // Citation refs: [1], [1, 2], [1,2,3]
    if (withCitations && ch === "[") {
      const end = text.indexOf("]", i + 1);
      if (end !== -1) {
        const inner = text.slice(i + 1, end);
        if (/^[\s\d,]+$/.test(inner)) {
          const refs = inner.split(",").map((s) => parseInt(s.trim(), 10) - 1).filter((n) => !isNaN(n) && n >= 0);
          if (refs.length > 0) { flush(); tokens.push({ t: "cite", refs }); i = end + 1; continue; }
        }
      }
    }

    buf += ch;
    i++;
  }
  flush();
  return tokens;
}

// ──────────────────────────────────────────────────────────────
// Inline renderer
// ──────────────────────────────────────────────────────────────

export function renderInlineTokens(
  tokens: InlineToken[],
  onCitationClick?: (ref: number) => void,
): React.ReactNode[] {
  return tokens.flatMap((tok, i) => {
    switch (tok.t) {
      case "bold":
        return [<strong key={i} className="font-semibold text-text">{tok.v}</strong>];
      case "italic":
        return [<em key={i} className="italic text-text/90">{tok.v}</em>];
      case "bold_italic":
        return [<strong key={i} className="font-semibold italic text-text">{tok.v}</strong>];
      case "code":
        return [<code key={i} className="rounded bg-slate-100 px-1 py-px text-[11px] font-mono text-rose-700 border border-slate-200/80">{tok.v}</code>];
      case "math":
        return [
          <span key={i} className="inline-flex items-baseline rounded bg-teal-50 px-1.5 py-px font-mono text-[11px] italic text-teal-800 border border-teal-200/60 mx-0.5">
            {tok.v}
          </span>,
        ];
      case "cite":
        return tok.refs.map((ref, ri) =>
          onCitationClick ? (
            <button
              key={`${i}-${ri}`}
              onClick={() => onCitationClick(ref)}
              className="mx-1 inline-flex h-4 min-w-[20px] items-center justify-center rounded bg-primary/10 px-1 text-[10px] font-bold text-primary hover:bg-primary/20 transition"
            >
              [{ref + 1}]
            </button>
          ) : (
            <sup key={`${i}-${ri}`} className="font-bold text-primary text-[9px] mx-0.5">[{ref + 1}]</sup>
          )
        );
      default:
        return [<span key={i}>{tok.v}</span>];
    }
  });
}

// ──────────────────────────────────────────────────────────────
// Block-level types
// ──────────────────────────────────────────────────────────────

type Block =
  | { type: "heading"; level: 1 | 2 | 3 | 4; text: string }
  | { type: "paragraph"; lines: string[] }
  | { type: "code_block"; lang: string; code: string }
  | { type: "math_block"; math: string }
  | { type: "blockquote"; lines: string[] }
  | { type: "ul"; items: string[][] }  // nested items as line arrays
  | { type: "ol"; items: string[][] }
  | { type: "hr" }
  | { type: "table_raw"; raw: string }
  | { type: "image"; alt: string; url: string };

const IMAGE_LINE_RE = /^!\[([^\]]*)\]\(([^)\s]+)\)\s*$/;

// ──────────────────────────────────────────────────────────────
// Block parser
// ──────────────────────────────────────────────────────────────

export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Empty line
    if (!trimmed) { i++; continue; }

    // Fenced code block: ``` or ~~~
    if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
      const fence = trimmed.slice(0, 3);
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence)) { codeLines.push(lines[i]); i++; }
      i++; // skip closing fence
      blocks.push({ type: "code_block", lang, code: codeLines.join("\n") });
      continue;
    }

    // Math block: $$ on its own line or $$...$$ inline block
    if (trimmed === "$$") {
      const mathLines: string[] = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") { mathLines.push(lines[i]); i++; }
      i++;
      blocks.push({ type: "math_block", math: mathLines.join("\n") });
      continue;
    }
    if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
      blocks.push({ type: "math_block", math: trimmed.slice(2, -2).trim() });
      i++;
      continue;
    }

    // Horizontal rule
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }

    // Heading: # text
    const hm = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hm) {
      blocks.push({ type: "heading", level: Math.min(4, hm[1].length) as 1 | 2 | 3 | 4, text: hm[2] });
      i++;
      continue;
    }

    // Setext heading (underline style): line followed by === or ---
    if (i + 1 < lines.length) {
      const next = lines[i + 1].trim();
      if (/^=+$/.test(next)) { blocks.push({ type: "heading", level: 1, text: trimmed }); i += 2; continue; }
      if (/^-+$/.test(next) && next.length >= 3) { blocks.push({ type: "heading", level: 2, text: trimmed }); i += 2; continue; }
    }

    // Blockquote: > text
    if (trimmed.startsWith(">")) {
      const qLines: string[] = [trimmed.replace(/^>\s?/, "")];
      i++;
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        qLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      blocks.push({ type: "blockquote", lines: qLines });
      continue;
    }

    // Table: starts with |
    if (trimmed.startsWith("|")) {
      const tableLines: string[] = [line];
      i++;
      while (i < lines.length && lines[i].trim().startsWith("|")) { tableLines.push(lines[i]); i++; }
      blocks.push({ type: "table_raw", raw: tableLines.join("\n") });
      continue;
    }

    // Unordered list: - / * / +
    if (/^[-*+]\s/.test(trimmed)) {
      const items: string[][] = [[trimmed.replace(/^[-*+]\s+/, "")]];
      i++;
      while (i < lines.length && /^[-*+]\s/.test(lines[i].trim())) {
        items.push([lines[i].trim().replace(/^[-*+]\s+/, "")]);
        i++;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    // Ordered list: 1. / 1)
    if (/^\d+[.)]\s/.test(trimmed)) {
      const items: string[][] = [[trimmed.replace(/^\d+[.)]\s+/, "")]];
      i++;
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i].trim())) {
        items.push([lines[i].trim().replace(/^\d+[.)]\s+/, "")]);
        i++;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    // Image: ![alt](url) on its own line
    const im = trimmed.match(IMAGE_LINE_RE);
    if (im) {
      blocks.push({ type: "image", alt: im[1], url: im[2] });
      i++;
      continue;
    }

    // Paragraph: collect consecutive non-special lines
    const paraLines: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^#{1,4}\s/.test(lines[i].trim()) &&
      !lines[i].trim().startsWith(">") &&
      !lines[i].trim().startsWith("|") &&
      !/^[-*+]\s/.test(lines[i].trim()) &&
      !/^\d+[.)]\s/.test(lines[i].trim()) &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith("~~~") &&
      !lines[i].trim().startsWith("$$") &&
      !IMAGE_LINE_RE.test(lines[i].trim()) &&
      !/^(-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    blocks.push({ type: "paragraph", lines: paraLines });
  }

  return blocks;
}

// ──────────────────────────────────────────────────────────────
// Main MarkdownRenderer component
// ──────────────────────────────────────────────────────────────

type MarkdownRendererProps = {
  text: string;
  className?: string;
  /** When provided, citation refs [N] become clickable buttons */
  onCitationClick?: (ref: number) => void;
  /** Compact mode reduces vertical spacing (e.g., evidence snippets) */
  compact?: boolean;
};

export default function MarkdownRenderer({
  text,
  className = "",
  onCitationClick,
  compact = false,
}: MarkdownRendererProps) {
  const blocks = parseBlocks(text);
  const gap = compact ? "space-y-1.5" : "space-y-3";

  function renderInline(t: string) {
    return renderInlineTokens(parseInline(t, !!onCitationClick), onCitationClick);
  }

  return (
    <div className={`${gap} ${className}`}>
      {blocks.map((block, bi) => {
        switch (block.type) {
          case "heading": {
            const level = block.level;
            const headingClass = [
              "font-heading font-bold text-text leading-snug",
              level === 1 && (compact ? "text-base mt-1" : "text-lg mt-3"),
              level === 2 && (compact ? "text-sm mt-1" : "text-base mt-2"),
              level === 3 && (compact ? "text-xs mt-1 uppercase tracking-wide" : "text-sm mt-1.5"),
              level === 4 && "text-[11px] uppercase tracking-widest text-muted mt-1",
            ].filter(Boolean).join(" ");
            return React.createElement(
              `h${level}` as "h1" | "h2" | "h3" | "h4",
              { key: bi, className: headingClass },
              renderInline(block.text)
            );
          }

          case "paragraph":
            return (
              <p key={bi} className="text-sm leading-relaxed text-text">
                {block.lines.map((line, li) => (
                  <React.Fragment key={li}>
                    {li > 0 && <br />}
                    {renderInline(line)}
                  </React.Fragment>
                ))}
              </p>
            );

          case "code_block":
            return (
              <div key={bi} className="overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
                {block.lang && (
                  <div className="border-b border-slate-700/80 bg-slate-800 px-3 py-1 text-[10px] font-mono font-semibold uppercase tracking-wider text-slate-400">
                    {block.lang}
                  </div>
                )}
                <pre className="overflow-x-auto px-4 py-3 text-[12px] font-mono leading-5 text-slate-100">
                  <code>{block.code}</code>
                </pre>
              </div>
            );

          case "math_block":
            return (
              <div key={bi} className="rounded-lg border border-teal-200 bg-teal-50 px-4 py-3">
                <div className="mb-1.5 flex items-center gap-1.5">
                  <span className="inline-flex items-center gap-1 rounded bg-teal-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-teal-700">
                    ∑ Formula
                  </span>
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-sm italic leading-relaxed text-teal-900">
                  {block.math}
                </pre>
              </div>
            );

          case "blockquote":
            return (
              <blockquote key={bi} className="rounded-r-md border-l-[3px] border-secondary bg-teal-50/60 py-2 pl-4 pr-2">
                {block.lines.map((line, li) => (
                  <p key={li} className="text-sm italic leading-relaxed text-muted">
                    {renderInline(line)}
                  </p>
                ))}
              </blockquote>
            );

          case "ul":
            return (
              <ul key={bi} className={compact ? "space-y-0.5 pl-3" : "space-y-1 pl-4"}>
                {block.items.map((itemLines, ii) => (
                  <li key={ii} className="flex items-start gap-2 text-sm leading-relaxed text-text">
                    <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-primary/60" />
                    <span className="flex-1">{itemLines.map((l, li2) => <React.Fragment key={li2}>{li2 > 0 && <br />}{renderInline(l)}</React.Fragment>)}</span>
                  </li>
                ))}
              </ul>
            );

          case "ol":
            return (
              <ol key={bi} className={compact ? "space-y-0.5 pl-3" : "space-y-1.5 pl-4"}>
                {block.items.map((itemLines, ii) => (
                  <li key={ii} className="flex items-start gap-2 text-sm leading-relaxed text-text">
                    <span className="mt-0.5 min-w-[1.25rem] shrink-0 text-right text-xs font-bold text-primary/70">
                      {ii + 1}.
                    </span>
                    <span className="flex-1">{itemLines.map((l, li2) => <React.Fragment key={li2}>{li2 > 0 && <br />}{renderInline(l)}</React.Fragment>)}</span>
                  </li>
                ))}
              </ol>
            );

          case "hr":
            return <hr key={bi} className="border-outline" />;

          case "image":
            return (
              <figure key={bi} className="my-2 overflow-hidden rounded-xl border border-outline bg-surface-low">
                <img
                  src={resolveImageUrl(block.url)}
                  alt={block.alt}
                  loading="lazy"
                  className="block max-h-[420px] w-full object-contain bg-white"
                  onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                />
                {block.alt && (
                  <figcaption className="border-t border-outline/40 px-3 py-1.5 text-[11px] italic text-muted">
                    {block.alt}
                  </figcaption>
                )}
              </figure>
            );

          case "table_raw": {
            const parsed = parseMarkdownTable(block.raw);
            if (parsed) return <TableView key={bi} table={parsed} />;
            return <pre key={bi} className="overflow-x-auto rounded bg-slate-50 p-2 text-xs text-muted">{block.raw}</pre>;
          }

          default:
            return null;
        }
      })}
    </div>
  );
}
