import { useState, useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { HistoryItem } from '@/types'
import { Activity, CalendarDays } from 'lucide-react'

interface ScanActivityTrendsProps {
  items: HistoryItem[]
  loading?: boolean
}

type Granularity = 'daily' | 'weekly' | 'monthly'

export default function ScanActivityTrends({
  items,
  loading = false,
}: ScanActivityTrendsProps) {
  const [granularity, setGranularity] = useState<Granularity>('daily')

  // Aggregate time series
  const trendData = useMemo(() => {
    if (!items || items.length === 0) return []

    // Map: bucketKey -> { label, count, timestamp }
    const buckets: Record<string, { label: string; count: number; sortKey: number }> = {}

    for (const item of items) {
      if (!item.created_at) continue
      const date = new Date(item.created_at)
      if (isNaN(date.getTime())) continue

      let bucketKey = ''
      let label = ''
      let sortKey = 0

      if (granularity === 'daily') {
        const year = date.getFullYear()
        const month = String(date.getMonth() + 1).padStart(2, '0')
        const day = String(date.getDate()).padStart(2, '0')
        bucketKey = `${year}-${month}-${day}`
        label = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
        sortKey = new Date(year, date.getMonth(), date.getDate()).getTime()
      } else if (granularity === 'weekly') {
        // Week starting Sunday
        const d = new Date(date)
        const day = d.getDay()
        const diff = d.getDate() - day
        const sunday = new Date(d.setDate(diff))
        const year = sunday.getFullYear()
        const month = String(sunday.getMonth() + 1).padStart(2, '0')
        const sDay = String(sunday.getDate()).padStart(2, '0')
        bucketKey = `${year}-W-${month}-${sDay}`
        label = `Wk of ${sunday.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`
        sortKey = sunday.getTime()
      } else {
        // Monthly
        const year = date.getFullYear()
        const month = String(date.getMonth() + 1).padStart(2, '0')
        bucketKey = `${year}-${month}`
        label = date.toLocaleDateString(undefined, { month: 'short', year: '2-digit' })
        sortKey = new Date(year, date.getMonth(), 1).getTime()
      }

      if (!buckets[bucketKey]) {
        buckets[bucketKey] = { label, count: 0, sortKey }
      }
      buckets[bucketKey].count += 1
    }

    const sorted = Object.values(buckets).sort((a, b) => a.sortKey - b.sortKey)
    return sorted
  }, [items, granularity])

  // Peak activity
  const peakVolume = useMemo(() => {
    if (!trendData.length) return 0
    return Math.max(...trendData.map(d => d.count))
  }, [trendData])

  const avgPerPeriod = useMemo(() => {
    if (!trendData.length) return 0
    const total = trendData.reduce((acc, d) => acc + d.count, 0)
    return (total / trendData.length).toFixed(1)
  }, [trendData])

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
      {/* Header & Granularity switch */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-emerald-600" />
            <h3 className="font-heading font-bold text-base text-slate-900">
              Scan Activity Trends
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Temporal distribution of user uploads and automated classifications
          </p>
        </div>

        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl text-xs self-start sm:self-auto">
          <button
            onClick={() => setGranularity('daily')}
            className={`px-3 py-1 rounded-lg font-medium transition ${
              granularity === 'daily'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Daily
          </button>
          <button
            onClick={() => setGranularity('weekly')}
            className={`px-3 py-1 rounded-lg font-medium transition ${
              granularity === 'weekly'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Weekly
          </button>
          <button
            onClick={() => setGranularity('monthly')}
            className={`px-3 py-1 rounded-lg font-medium transition ${
              granularity === 'monthly'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Monthly
          </button>
        </div>
      </div>

      {/* Quick stats pills */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/60">
          <span className="text-slate-500 block text-[11px]">Total Timeline Scans</span>
          <span className="text-base font-extrabold text-slate-900">{items.length}</span>
        </div>
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/60">
          <span className="text-slate-500 block text-[11px]">Avg / {granularity.slice(0, -2)}</span>
          <span className="text-base font-extrabold text-emerald-700">{avgPerPeriod}</span>
        </div>
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/60 col-span-2 sm:col-span-1">
          <span className="text-slate-500 block text-[11px]">Peak Interval Volume</span>
          <span className="text-base font-extrabold text-blue-700">{peakVolume} scans</span>
        </div>
      </div>

      {/* Chart Canvas */}
      {loading ? (
        <div className="h-64 flex items-center justify-center text-sm text-slate-400">
          Loading timeline data…
        </div>
      ) : trendData.length === 0 ? (
        <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-slate-400 text-xs">
          <CalendarDays className="h-8 w-8 text-slate-300 mb-2" />
          <p>No activity records available for this period.</p>
        </div>
      ) : (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={trendData}
              margin={{ top: 10, right: 10, left: -20, bottom: 5 }}
            >
              <defs>
                <linearGradient id="scanGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 11, fill: '#64748B' }}
                tickMargin={8}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748B' }}
                allowDecimals={false}
              />
              <Tooltip
                formatter={(val: number) => [`${val} scans`, 'Volume']}
                labelFormatter={(label: string) => `Period: ${label}`}
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  backgroundColor: '#FFFFFF',
                  fontSize: '12px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                }}
              />
              <Area
                type="monotone"
                dataKey="count"
                stroke="#10B981"
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#scanGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
