import type { ModelMetrics } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'

interface Props {
  metrics: ModelMetrics
}

export default function ConfusionMatrixView({ metrics }: Props) {
  const { confusion_matrix: cm, class_names } = metrics
  const maxVal = Math.max(...cm.flat(), 1)

  return (
    <div className="card-eco overflow-hidden">
      <div className="section-header">
        <div>
          <h2 className="section-title">Confusion Matrix Heatmap</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Diagonal intensity indicates true positive agreement; off-diagonal rose cells indicate misclassifications.
          </p>
        </div>
      </div>
      <div className="p-5 overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs mx-auto">
          <thead>
            <tr>
              <th className="p-2 text-slate-400 text-right pr-3 font-semibold uppercase tracking-wider text-[10px]">
                True \ Pred
              </th>
              {class_names.map(c => (
                <th
                  key={c}
                  className="p-2 text-center font-bold whitespace-nowrap"
                  style={{ color: CATEGORY_COLORS[c] ?? '#334155' }}
                >
                  {capitalize(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={class_names[i]}>
                <td
                  className="p-2 text-right font-bold pr-3 whitespace-nowrap"
                  style={{ color: CATEGORY_COLORS[class_names[i]] ?? '#334155' }}
                >
                  {capitalize(class_names[i])}
                </td>
                {row.map((val, j) => {
                  const intensity = maxVal > 0 ? val / maxVal : 0
                  const isDiag = i === j
                  const bg = isDiag
                    ? `rgba(22, 163, 74, ${0.12 + intensity * 0.72})`
                    : val > 0
                      ? `rgba(244, 63, 94, ${0.12 + intensity * 0.55})`
                      : '#F8FAFC'
                  const textClass =
                    isDiag && intensity > 0.5
                      ? 'text-white font-bold'
                      : !isDiag && val > 0
                        ? 'text-rose-900 font-semibold'
                        : 'text-slate-400'
                  return (
                    <td
                      key={j}
                      className={`p-2.5 text-center rounded-lg font-mono border border-slate-200/40 ${textClass}`}
                      style={{ background: bg, minWidth: '54px' }}
                      title={`True: ${capitalize(class_names[i])} → Predicted: ${capitalize(class_names[j])} (${val})`}
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
