import MarkdownRenderer, { parseMarkdownTable, TableView } from "./MarkdownRenderer";

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

export default function SnippetRenderer({
  text,
  blockType,
  maxRows,
  textClassName = "",
  compact = false,
}: SnippetRendererProps) {
  // Table detection: explicit block type or markdown table syntax
  const isTable = blockType === "table" || text.trimStart().startsWith("|");
  if (isTable) {
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
