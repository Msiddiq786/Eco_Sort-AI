import type { ModelMetrics } from '@/types'
import CategoryBadge from '@/components/ui/CategoryBadge'

interface Props {
  metrics: ModelMetrics
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card text-center items-center justify-center">
      <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">{label}</p>
      <p className="text-2xl sm:text-3xl font-heading font-extrabold text-ecosort-emerald">{value}</p>
    </div>
  )
}

export default function MetricsOverview({ metrics }: Props) {
  const pct = (v: number) => `${(v * 100).toFixed(2)}%`
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricBox label="Test Accuracy" value={pct(metrics.accuracy)} />
        <MetricBox label="Macro Precision" value={pct(metrics.macro_precision)} />
        <MetricBox label="Macro Recall" value={pct(metrics.macro_recall)} />
        <MetricBox label="Macro F1 Score" value={pct(metrics.macro_f1)} />
      </div>

      <div className="card-eco overflow-hidden">
        <div className="section-header">
          <h3 className="section-title">Per-Class Evaluation Breakdown</h3>
          <span className="text-xs text-slate-500">{metrics.class_names.length} evaluated classes</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50/80 border-b border-slate-200 text-xs uppercase text-slate-500">
              <tr>
                {['Class', 'Precision', 'Recall', 'F1 Score', 'Support'].map(h => (
                  <th key={h} className="text-left px-5 py-3 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {metrics.class_names.map(cls => {
                const m = metrics.per_class[cls]
                return (
                  <tr key={cls} className="hover:bg-slate-50/70 transition-colors">
                    <td className="px-5 py-3">
                      <CategoryBadge category={cls} size="sm" />
                    </td>
                    <td className="px-5 py-3 font-mono font-semibold text-slate-700">{pct(m.precision)}</td>
                    <td className="px-5 py-3 font-mono font-semibold text-slate-700">{pct(m.recall)}</td>
                    <td className="px-5 py-3 font-mono font-bold text-ecosort-emerald">{pct(m.f1)}</td>
                    <td className="px-5 py-3 font-mono text-slate-500">{m.support}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
