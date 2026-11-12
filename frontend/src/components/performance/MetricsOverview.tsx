import type { ModelMetrics } from '@/types'
import { capitalize } from '@/lib/utils'

interface Props { metrics: ModelMetrics }

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4 text-center">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className="text-2xl font-bold text-primary">{value}</p>
    </div>
  )
}

export default function MetricsOverview({ metrics }: Props) {
  const pct = (v: number) => `${(v * 100).toFixed(2)}%`
  return (
    <div className="space-y-4">
      <h2 className="font-semibold">Overall Metrics (Test Set)</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricBox label="Accuracy" value={pct(metrics.accuracy)} />
        <MetricBox label="Macro Precision" value={pct(metrics.macro_precision)} />
        <MetricBox label="Macro Recall" value={pct(metrics.macro_recall)} />
        <MetricBox label="Macro F1" value={pct(metrics.macro_f1)} />
      </div>

      <h3 className="font-semibold text-sm mt-4">Per-Class Metrics</h3>
      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              {['Class', 'Precision', 'Recall', 'F1', 'Support'].map(h => (
                <th key={h} className="text-left px-4 py-2 text-muted-foreground font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {metrics.class_names.map(cls => {
              const m = metrics.per_class[cls]
              return (
                <tr key={cls} className="hover:bg-muted/30">
                  <td className="px-4 py-2 font-medium">{capitalize(cls)}</td>
                  <td className="px-4 py-2">{pct(m.precision)}</td>
                  <td className="px-4 py-2">{pct(m.recall)}</td>
                  <td className="px-4 py-2">{pct(m.f1)}</td>
                  <td className="px-4 py-2 text-muted-foreground">{m.support}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
