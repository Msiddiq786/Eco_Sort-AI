import { useEffect, useState } from 'react'
import { getHistory } from '@/services/api'
import type { HistoryItem } from '@/types'
import { capitalize, formatPct, CATEGORY_COLORS } from '@/lib/utils'

export default function RecentTable() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getHistory({ page: 1, page_size: 6 })
      .then(d => setItems(d.items))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="rounded-xl border border-border/80 bg-white p-5 shadow-sm">
      <h2 className="font-heading font-semibold text-base text-slate-900 mb-4">
        Recent Classifications
      </h2>
      {loading ? (
        <p className="text-sm text-muted-foreground py-10 text-center">Loading recent records…</p>
      ) : items.length === 0 ? (
        <div className="py-12 text-center">
          <p className="text-sm text-muted-foreground">No recent classifications logged yet.</p>
          <p className="text-xs text-muted-foreground/80 mt-1">
            New classifications will appear here automatically.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-slate-100">
          {items.map(item => {
            const color = CATEGORY_COLORS[item.predicted_category] ?? '#888'
            return (
              <div
                key={item.id}
                className="flex items-center justify-between py-2.5 first:pt-0 last:pb-0"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ background: color }}
                  />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-800 truncate">
                      {capitalize(item.predicted_category)}
                    </p>
                    <p className="text-xs text-muted-foreground font-mono truncate max-w-[160px]">
                      {item.filename}
                    </p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-sm font-semibold text-emerald-700">
                    {formatPct(item.confidence)}
                  </span>
                  <p className="text-[11px] text-muted-foreground">
                    {new Date(item.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
