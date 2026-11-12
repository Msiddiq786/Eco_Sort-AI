import type { HistoryItem, WasteCategory } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib'
import { Layers, CheckCircle2, TrendingUp, Sparkles } from 'lucide-react'

interface HistorySummaryCardsProps {
  items: HistoryItem[]
  totalServerRecords: number
}

export default function HistorySummaryCards({ items, totalServerRecords }: HistorySummaryCardsProps) {
  if (!items.length) return null

  // Category counts in current filtered set
  const counts: Record<string, number> = {}
  let highConfCount = 0
  let totalConfidence = 0

  for (const item of items) {
    counts[item.predicted_category] = (counts[item.predicted_category] || 0) + 1
    if (item.confidence >= 0.70) highConfCount += 1
    totalConfidence += item.confidence
  }

  const avgConfidence = items.length > 0 ? (totalConfidence / items.length) * 100 : 0
  const highConfPct = items.length > 0 ? (highConfCount / items.length) * 100 : 0

  // Find top category
  let topCategory: WasteCategory = 'plastic'
  let topCount = 0
  for (const [cat, cnt] of Object.entries(counts)) {
    if (cnt > topCount) {
      topCount = cnt
      topCategory = cat as WasteCategory
    }
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Total Classified</span>
          <Layers className="h-4 w-4 text-emerald-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-slate-800">
          {totalServerRecords}
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          {items.length} records in active view
        </p>
      </div>

      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Top Waste Material</span>
          <TrendingUp className="h-4 w-4 text-blue-600" />
        </div>
        <div className="flex items-center gap-2">
          <span
            className="h-3 w-3 rounded-full"
            style={{ background: CATEGORY_COLORS[topCategory] ?? '#888' }}
          />
          <span className="text-2xl font-heading font-bold text-slate-800">
            {capitalize(topCategory)}
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          {topCount} items ({Math.round((topCount / items.length) * 100)}% of viewed)
        </p>
      </div>

      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Avg. Model Confidence</span>
          <Sparkles className="h-4 w-4 text-amber-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-slate-800">
          {avgConfidence.toFixed(1)}%
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Dual-model ensemble metric
        </p>
      </div>

      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>High Certainty Rate</span>
          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-slate-800">
          {highConfPct.toFixed(0)}%
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          {highConfCount} of {items.length} &ge; 70% confidence
        </p>
      </div>
    </div>
  )
}
