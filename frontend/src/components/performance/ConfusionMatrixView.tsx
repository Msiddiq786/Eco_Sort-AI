import type { ModelMetrics } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'

interface Props { metrics: ModelMetrics }

export default function ConfusionMatrixView({ metrics }: Props) {
  const { confusion_matrix: cm, class_names } = metrics
  const maxVal = Math.max(...cm.flat())

  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <h2 className="font-semibold text-sm mb-4">Confusion Matrix</h2>
      <div className="overflow-x-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="p-1 text-muted-foreground text-right pr-2">True \ Pred</th>
              {class_names.map(c => (
                <th key={c} className="p-1 text-center font-medium" style={{ color: CATEGORY_COLORS[c] }}>
                  {capitalize(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={class_names[i]}>
                <td className="p-1 text-right font-medium pr-2" style={{ color: CATEGORY_COLORS[class_names[i]] }}>
                  {capitalize(class_names[i])}
                </td>
                {row.map((val, j) => {
                  const intensity = maxVal > 0 ? val / maxVal : 0
                  const bg = i === j
                    ? `rgba(34,197,94,${0.15 + intensity * 0.6})`
                    : val > 0 ? `rgba(239,68,68,${0.1 + intensity * 0.5})` : 'transparent'
                  return (
                    <td
                      key={j}
                      className="p-2 text-center rounded font-mono"
                      style={{ background: bg, minWidth: '48px' }}
                    >
                      {val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
