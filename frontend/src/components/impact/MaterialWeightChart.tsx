import type { WasteCategory, WasteWeightLog } from '@/types'
import { capitalize, CATEGORY_COLORS, WASTE_CLASSES } from '@/lib'
import { Layers } from 'lucide-react'

interface MaterialWeightChartProps {
  logs: WasteWeightLog[]
}

export default function MaterialWeightChart({ logs }: MaterialWeightChartProps) {
  // Compute category weights
  const weights: Record<WasteCategory, number> = {
    cardboard: 0,
    glass: 0,
    metal: 0,
    paper: 0,
    plastic: 0,
    trash: 0,
  }

  let totalGrams = 0
  for (const log of logs) {
    weights[log.category] = (weights[log.category] || 0) + log.weightGrams
    totalGrams += log.weightGrams
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-emerald-600" />
          <h3 className="font-heading font-bold text-base text-slate-800">
            Diverted Material Breakdown
          </h3>
        </div>
        <span className="text-xs text-muted-foreground font-medium">
          Total Logged: {(totalGrams / 1000).toFixed(2)} kg
        </span>
      </div>

      <div className="space-y-3">
        {WASTE_CLASSES.map(cls => {
          const grams = weights[cls]
          const kg = (grams / 1000).toFixed(2)
          const pct = totalGrams > 0 ? Math.round((grams / totalGrams) * 100) : 0
          const color = CATEGORY_COLORS[cls] ?? '#888'

          return (
            <div key={cls} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: color }}
                  />
                  <span className="font-semibold text-slate-800">{capitalize(cls)}</span>
                </div>
                <div className="flex items-center gap-2 text-slate-600">
                  <span className="font-mono">{kg} kg</span>
                  <span className="text-muted-foreground font-semibold">({pct}%)</span>
                </div>
              </div>

              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.max(1, pct)}%`,
                    background: color,
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
