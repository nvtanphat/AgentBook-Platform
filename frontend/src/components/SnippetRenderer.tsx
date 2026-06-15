import MarkdownRenderer, { parseMarkdownTable, TableView, type ParsedTable } from "./MarkdownRenderer";

type SnippetRendererProps = {
  text: string;
  blockType?: string;
  /** Max rows for table preview. Undefined = show all. */
  maxRows?: number;
  /** CSS class applied to the wrapper */
  textClassName?: string;
  /** Use compact spacing (evidence panels) */
  compact?: boolean;
};

function extractTagContent(html: string, tag: string): string[] {
  const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "gi");
  const matches: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    matches.push(m[1].replace(/<[^>]+>/g, "").trim());
  }
  return matches;
}

function parseHtmlTable(html: string): ParsedTable | null {
  const thead = /<thead[\s\S]*?<\/thead>/i.exec(html)?.[0] ?? "";
  const headers = thead
    ? extractTagContent(thead, "th")
    : extractTagContent(/<tr[^>]*>[\s\S]*?<\/tr>/i.exec(html)?.[0] ?? "", "th");

  const tbodyMatch = /<tbody[\s\S]*?<\/tbody>/i.exec(html);
  const rowsHtml = tbodyMatch
    ? tbodyMatch[0]
    : html.replace(/<thead[\s\S]*?<\/thead>/i, "");

  const trRe = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
  const rows: string[][] = [];
  let tr: RegExpExecArray | null;
  while ((tr = trRe.exec(rowsHtml)) !== null) {
    const cells = extractTagContent(tr[0], "td");
    if (cells.length > 0) rows.push(cells);
  }

  if (rows.length === 0 && headers.length === 0) return null;
  return { headers, rows };
}

export default function SnippetRenderer({
  text,
  blockType,
  maxRows,
  textClassName = "",
  compact = false,
}: SnippetRendererProps) {
  const trimmed = text.trimStart();

  // Raw HTML table from spreadsheet/table blocks
  if (blockType === "table" && trimmed.toLowerCase().startsWith("<table")) {
    const parsed = parseHtmlTable(trimmed);
    if (parsed) return <TableView table={parsed} maxRows={maxRows} className={textClassName} />;
  }

  // Markdown pipe-table syntax
  const isMarkdownTable = blockType === "table" || trimmed.startsWith("|");
  if (isMarkdownTable) {
    const parsed = parseMarkdownTable(text);
    if (parsed) return <TableView table={parsed} maxRows={maxRows} className={textClassName} />;
  }

  // Render rich markdown for all other content
  return (
    <MarkdownRenderer
      text={text}
      compact={compact}
      className={textClassName}
    />
  );
}
