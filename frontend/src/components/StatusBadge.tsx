import { CheckCircle2, AlertCircle, Loader2, Upload, ScanText, Layers, Cpu } from "lucide-react";

type StageConfig = {
  style: string;
  label: string;
  icon: React.ReactNode;
};

const STAGE_CONFIG: Record<string, StageConfig> = {
  uploaded: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Uploaded",
    icon: <Upload size={9} />,
  },
  parsing: {
    style: "bg-sky-50 text-sky-700 border-sky-200",
    label: "Parsing…",
    icon: <ScanText size={9} className="animate-pulse" />,
  },
  parsed: {
    style: "bg-indigo-50 text-indigo-700 border-indigo-200",
    label: "Parsed",
    icon: <Layers size={9} />,
  },
  indexing: {
    style: "bg-amber-50 text-amber-700 border-amber-200",
    label: "Indexing…",
    icon: <Cpu size={9} className="animate-pulse" />,
  },
  indexed: {
    style: "bg-emerald-50 text-emerald-700 border-emerald-200",
    label: "Ready",
    icon: <CheckCircle2 size={9} />,
  },
  failed: {
    style: "bg-red-50 text-red-700 border-red-200",
    label: "Failed",
    icon: <AlertCircle size={9} />,
  },
};

const FALLBACK: StageConfig = {
  style: "bg-slate-50 text-slate-600 border-slate-200",
  label: "",
  icon: <Loader2 size={9} className="animate-spin" />,
};

export default function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const cfg = STAGE_CONFIG[normalized] ?? { ...FALLBACK, label: normalized };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${cfg.style}`}>
      {cfg.icon}
      {cfg.label}
    </span>
  );
}
