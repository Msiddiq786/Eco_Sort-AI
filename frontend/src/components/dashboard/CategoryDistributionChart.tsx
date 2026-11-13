import { useState } from 'react'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import { BarChart3, PieChart as PieIcon } from 'lucide-react'

export interface CategoryDataPoint {
  category: string
  count: number
  percentage: number
  color: string
}

interface CategoryDistributionChartProps {
  distribution: CategoryDataPoint[]
  totalCount: number
  loading?: boolean
}

export default function CategoryDistributionChart({
  distribution,
  totalCount,
  loading = false,
}: CategoryDistributionChartProps) {
  const [chartType, setChartType] = useState<'donut' | 'bar'>('donut')

  const chartData = distribution.map(d => ({
    name: capitalize(d.category),
    categoryKey: d.category,
    value: d.count,
    percentage: d.percentage,
    color: d.color || CATEGORY_COLORS[d.category] || '#64748B',
  }))

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
      {/* Header & View Switcher */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-heading font-bold text-base text-slate-900">
            Category Distribution (7-Class)
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Breakdown of all {totalCount} classification scans
          </p>
        </div>
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl text-xs">
          <button
            onClick={() => setChartType('donut')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition ${
              chartType === 'donut'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <PieIcon className="h-3.5 w-3.5" />
            <span>Donut</span>
          </button>
          <button
            onClick={() => setChartType('bar')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition ${
              chartType === 'bar'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            <span>Bars</span>
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      {loading ? (
        <div className="h-64 flex items-center justify-center text-sm text-slate-400">
          Loading category distribution…
        </div>
      ) : totalCount === 0 ? (
        <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-slate-400 text-xs">
          <p>No scans recorded in the selected time range.</p>
          <p className="mt-1">Classify an image or adjust date filters to see distribution.</p>
        </div>
      ) : chartType === 'donut' ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={90}
                paddingAngle={3}
              >
                {chartData.map(entry => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value} scans (${totalCount > 0 ? ((value / totalCount) * 100).toFixed(1) : 0}%)`,
                  name,
                ]}
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  backgroundColor: '#FFFFFF',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                  fontSize: '12px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: '#64748B' }}
                interval={0}
                angle={-25}
                textAnchor="end"
              />
              <YAxis tick={{ fontSize: 11, fill: '#64748B' }} allowDecimals={false} />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value} scans (${totalCount > 0 ? ((value / totalCount) * 100).toFixed(1) : 0}%)`,
                  name,
                ]}
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  backgroundColor: '#FFFFFF',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {chartData.map(entry => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Legend & Breakdown Pills */}
      <div className="pt-2 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
        {chartData.map(d => (
          <div
            key={d.name}
            className="p-2 rounded-xl bg-slate-50 border border-slate-200/60 flex flex-col justify-between"
          >
            <div className="flex items-center gap-1.5 min-w-0">
              <span
                className="h-2 w-2 rounded-full shrink-0"
                style={{ background: d.color }}
              />
              <span className="font-semibold text-slate-700 truncate text-[11px]">
                {d.name}
              </span>
            </div>
            <div className="mt-1 flex items-baseline justify-between text-[11px]">
              <span className="font-bold text-slate-900">{d.value}</span>
              <span className="text-slate-500 font-mono text-[10px]">
                {d.percentage.toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
