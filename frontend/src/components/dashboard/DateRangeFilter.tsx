import { Calendar, Download, RefreshCw, SlidersHorizontal } from 'lucide-react'
import type { AnalyticsDateFilter } from '@/types'

interface DateRangeFilterProps {
  dateFilter: AnalyticsDateFilter
  onFilterChange: (filter: AnalyticsDateFilter) => void
  customStart: string
  customEnd: string
  onCustomStartChange: (val: string) => void
  onCustomEndChange: (val: string) => void
  totalFilteredCount: number
  onExportCsv: () => void
  onRefresh: () => void
  isRefreshing?: boolean
}

const PRESETS: { id: AnalyticsDateFilter; label: string }[] = [
  { id: 'today', label: 'Today' },
  { id: '7days', label: 'Last 7 Days' },
  { id: '30days', label: 'Last 30 Days' },
  { id: '90days', label: 'Last 90 Days' },
  { id: 'all', label: 'All Time' },
  { id: 'custom', label: 'Custom Range' },
]

export default function DateRangeFilter({
  dateFilter,
  onFilterChange,
  customStart,
  customEnd,
  onCustomStartChange,
  onCustomEndChange,
  totalFilteredCount,
  onExportCsv,
  onRefresh,
  isRefreshing = false,
}: DateRangeFilterProps) {
  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-4 sm:p-5 shadow-sm space-y-3.5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left: Presets bar */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 lg:pb-0 scrollbar-thin">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 mr-2 shrink-0">
            <Calendar className="h-4 w-4 text-emerald-600" />
            <span>Time Window:</span>
          </div>
          {PRESETS.map(({ id, label }) => {
            const isActive = dateFilter === id
            return (
              <button
                key={id}
                onClick={() => onFilterChange(id)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-emerald-600 text-white font-semibold shadow-sm'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200/70 hover:text-slate-900'
                }`}
              >
                {label}
              </button>
            )
          })}
        </div>

        {/* Right: Actions (Refresh & Export) */}
        <div className="flex items-center gap-2 self-end lg:self-auto">
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 text-xs font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition disabled:opacity-50"
            title="Refresh analytics data"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin text-emerald-600' : ''}`} />
            <span>Sync</span>
          </button>
          <button
            onClick={onExportCsv}
            disabled={totalFilteredCount === 0}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold transition shadow-sm disabled:opacity-50"
            title="Download CSV report of filtered scan data"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Custom Date Pickers Drawer */}
      {dateFilter === 'custom' && (
        <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-3 text-xs bg-slate-50/70 p-3 rounded-xl">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="h-3.5 w-3.5 text-slate-500" />
            <span className="font-semibold text-slate-700">Custom Date Range:</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-slate-500">From:</label>
            <input
              type="date"
              value={customStart}
              onChange={e => onCustomStartChange(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-800 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-slate-500">To:</label>
            <input
              type="date"
              value={customEnd}
              onChange={e => onCustomEndChange(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-800 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
            />
          </div>
          <span className="text-[11px] text-slate-400">
            (Filter applies automatically to all dashboard charts & metrics)
          </span>
        </div>
      )}
    </div>
  )
}
