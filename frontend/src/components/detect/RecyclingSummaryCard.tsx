import type { RecyclingSummary } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import {
  Recycle,
  CheckCircle2,
  AlertTriangle,
  Leaf,
  HelpCircle,
  ShieldAlert,
} from 'lucide-react'

interface Props {
  summary: RecyclingSummary
}

export default function RecyclingSummaryCard({ summary }: Props) {
  const { total_objects, category_counts, status_counts } = summary
  const sumStatus =
    (status_counts.recyclable || 0) +
    (status_counts.conditionally_recyclable || 0) +
    (status_counts.compostable || 0) +
    (status_counts.hazardous || 0) +
    (status_counts.not_standard || 0) +
    (status_counts.unknown || 0)
  const isConsistent = sumStatus === total_objects

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-emerald-600 text-white">
            <Recycle className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-heading font-extrabold text-slate-900">
              Multi-Object Recycling Summary
            </h3>
            <p className="text-xs text-slate-500">
              Total detected items:{' '}
              <strong className="text-slate-800 font-bold">{total_objects}</strong>
              {isConsistent && (
                <span className="ml-2 text-[11px] text-emerald-600 font-medium">
                  ✓ {sumStatus}/{total_objects} accounted
                </span>
              )}
            </p>
          </div>
        </div>
        <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 text-xs font-bold border border-emerald-200">
          {total_objects} Items Analyzed
        </span>
      </div>

      {/* Category breakdown pills */}
      <div className="space-y-1.5">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
          Detected Materials
        </span>
        <div className="flex flex-wrap gap-2 pt-0.5">
          {Object.entries(category_counts).map(([cat, count]) => (
            <div
              key={cat}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-semibold border shadow-xs"
              style={{
                backgroundColor: `${CATEGORY_COLORS[cat] ?? '#16A34A'}15`,
                borderColor: `${CATEGORY_COLORS[cat] ?? '#16A34A'}40`,
                color: CATEGORY_COLORS[cat] ?? '#14532D',
              }}
            >
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: CATEGORY_COLORS[cat] ?? '#16A34A' }}
              />
              <span>{capitalize(cat)}:</span>
              <strong className="font-extrabold">{count}</strong>
            </div>
          ))}
        </div>
      </div>

      {/* Disposal status breakdown grid */}
      <div className="space-y-1.5 pt-1">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
          Disposal Action Stream Breakdown
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 pt-1">
          {/* Recyclable */}
          <div className="p-3 rounded-xl bg-blue-50/70 border border-blue-100 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-blue-800">
              <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />
              <span>Recyclable</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-blue-900 mt-1">
              {status_counts.recyclable}
            </span>
          </div>

          {/* Conditionally Recyclable */}
          <div className="p-3 rounded-xl bg-emerald-50/70 border border-emerald-100 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-emerald-800">
              <Recycle className="h-3.5 w-3.5 text-emerald-600" />
              <span>Conditional</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-emerald-900 mt-1">
              {status_counts.conditionally_recyclable}
            </span>
          </div>

          {/* Compostable */}
          <div className="p-3 rounded-xl bg-lime-50/70 border border-lime-100 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-lime-800">
              <Leaf className="h-3.5 w-3.5 text-lime-600" />
              <span>Compostable</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-lime-900 mt-1">
              {status_counts.compostable}
            </span>
          </div>

          {/* Hazardous */}
          <div className="p-3 rounded-xl bg-rose-50/70 border border-rose-100 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-rose-800">
              <ShieldAlert className="h-3.5 w-3.5 text-rose-600" />
              <span>Hazardous</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-rose-900 mt-1">
              {status_counts.hazardous}
            </span>
          </div>

          {/* Not Standard */}
          <div className="p-3 rounded-xl bg-slate-100/70 border border-slate-200 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-slate-700">
              <AlertTriangle className="h-3.5 w-3.5 text-slate-500" />
              <span>Residual</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-slate-800 mt-1">
              {status_counts.not_standard}
            </span>
          </div>

          {/* Unknown */}
          <div className="p-3 rounded-xl bg-amber-50/70 border border-amber-100 flex flex-col justify-between">
            <div className="flex items-center gap-1 text-[11px] font-semibold text-amber-800">
              <HelpCircle className="h-3.5 w-3.5 text-amber-600" />
              <span>Review</span>
            </div>
            <span className="text-xl font-heading font-extrabold text-amber-900 mt-1">
              {status_counts.unknown}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
