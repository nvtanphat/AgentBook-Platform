import { Lightbulb, TrendingUp, Sparkles } from 'lucide-react';

interface ReasoningStep {
  step_type: 'retrieve' | 'traverse' | 'synthesize';
  entities: string[];
  relations: string[];
  confidence: number;
  description: string;
}

interface ReasoningTraceProps {
  steps: ReasoningStep[];
  onStepHover?: (entities: string[]) => void;
}

const STEP_ICONS = {
  retrieve: <TrendingUp size={14} className="text-slate-500" />,
  traverse: <Sparkles size={14} className="text-slate-500" />,
  synthesize: <Lightbulb size={14} className="text-slate-500" />,
};

const STEP_COLORS = {
  retrieve: 'bg-slate-50 border-slate-200 hover:bg-slate-100',
  traverse: 'bg-slate-50 border-slate-200 hover:bg-slate-100',
  synthesize: 'bg-slate-50 border-slate-200 hover:bg-slate-100',
};

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = 'text-slate-600';

  return (
    <span className={`text-[10px] font-bold tabular-nums ${color}`}>
      {pct}%
    </span>
  );
}

export default function ReasoningTrace({ steps, onStepHover }: ReasoningTraceProps) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-4 space-y-2 border-t border-outline pt-3">
      <div className="flex items-center gap-1.5 mb-2">
        <Lightbulb size={12} className="text-primary" />
        <p className="text-xs font-semibold text-muted">
          Cách tôi tìm câu trả lời:
        </p>
      </div>

      {steps.map((step, i) => (
        <div
          key={i}
          className={`flex items-start gap-2 p-2.5 rounded-lg border transition-all cursor-pointer ${
            STEP_COLORS[step.step_type]
          }`}
          onMouseEnter={() => onStepHover?.(step.entities)}
          onMouseLeave={() => onStepHover?.([])}
        >
          {/* Step number & icon */}
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="flex items-center justify-center w-5 h-5 rounded-full bg-white text-[10px] font-bold text-primary border border-outline">
              {i + 1}
            </span>
            {STEP_ICONS[step.step_type]}
          </div>

          {/* Description */}
          <div className="flex-1 min-w-0">
            <p className="text-xs text-text leading-relaxed">
              {step.description}
            </p>

            {/* Entities */}
            {step.entities.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1.5">
                {step.entities.slice(0, 5).map((entity, j) => (
                  <span
                    key={j}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-white border border-outline text-text"
                  >
                    {entity}
                  </span>
                ))}
                {step.entities.length > 5 && (
                  <span className="text-[10px] text-muted">
                    +{step.entities.length - 5} mục khác
                  </span>
                )}
              </div>
            )}

            {/* Relations */}
            {step.relations.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {step.relations.map((rel, j) => (
                  <span
                    key={j}
                    className="text-[9px] px-1.5 py-0.5 rounded bg-white/60 text-muted italic"
                  >
                    {rel.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Confidence */}
          <div className="shrink-0">
            <ConfidenceBadge value={step.confidence} />
          </div>
        </div>
      ))}
    </div>
  );
}
