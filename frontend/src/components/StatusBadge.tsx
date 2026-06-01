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
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Reading",
    icon: <ScanText size={9} className="animate-pulse" />,
  },
  parsed: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Read",
    icon: <Layers size={9} />,
  },
  chunking: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Analyzing",
    icon: <Layers size={9} className="animate-pulse" />,
  },
  embedding: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Preparing",
    icon: <Cpu size={9} className="animate-pulse" />,
  },
  indexing: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Preparing",
    icon: <Cpu size={9} className="animate-pulse" />,
  },
  indexed: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Ready",
    icon: <CheckCircle2 size={9} className="text-emerald-500" />,
  },
  failed: {
    style: "bg-slate-100 text-slate-600 border-slate-200",
    label: "Needs retry",
    icon: <AlertCircle size={9} className="text-red-500" />,
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
