import {
  ScanLine,
  TrendingUp,
  Recycle,
  Apple,
  AlertTriangle,
  Percent,
} from 'lucide-react'
import { capitalize, formatPct } from '@/lib/utils'

export interface SummaryMetrics {
  totalScans: number
  mostDetectedCategory: {
    category: string
    count: number
    percentage: number
  } | null
  recyclableCount: number
  recyclablePercentage: number
  compostableCount: number
  compostablePercentage: number
  specialDisposalCount: number
  specialDisposalPercentage: number
  averageConfidence: number
}

interface AnalyticsSummaryCardsProps {
  metrics: SummaryMetrics
  loading?: boolean
}

export default function AnalyticsSummaryCards({
  metrics,
  loading = false,
}: AnalyticsSummaryCardsProps) {
  const cards = [
    {
      title: 'Total Scans',
      value: loading ? '—' : metrics.totalScans.toLocaleString(),
      subtext: 'Classification queries analyzed',
      icon: ScanLine,
      iconBadge: 'bg-emerald-50 text-ecosort-emerald border-emerald-200/70',
    },
    {
      title: 'Most Detected Class',
      value: loading
        ? '—'
        : metrics.mostDetectedCategory
        ? capitalize(metrics.mostDetectedCategory.category)
        : 'None',
      subtext: loading
        ? 'Calculating…'
        : metrics.mostDetectedCategory
        ? `${metrics.mostDetectedCategory.count} items (${metrics.mostDetectedCategory.percentage.toFixed(1)}% of total)`
        : 'No records in window',
      icon: TrendingUp,
      iconBadge: 'bg-sky-50 text-sky-600 border-sky-200/70',
    },
    {
      title: 'Recyclable Items',
      value: loading ? '—' : metrics.recyclableCount.toLocaleString(),
      subtext: loading
        ? 'Calculating…'
        : `${metrics.recyclablePercentage.toFixed(1)}% dry recyclable stream`,
      icon: Recycle,
      iconBadge: 'bg-teal-50 text-teal-600 border-teal-200/70',
    },
    {
      title: 'Compostable Items',
      value: loading ? '—' : metrics.compostableCount.toLocaleString(),
      subtext: loading
        ? 'Calculating…'
        : `${metrics.compostablePercentage.toFixed(1)}% organic wet waste`,
      icon: Apple,
      iconBadge: 'bg-lime-50 text-lime-700 border-lime-200/70',
    },
    {
      title: 'Residual / Review',
      value: loading ? '—' : metrics.specialDisposalCount.toLocaleString(),
      subtext: loading
        ? 'Calculating…'
        : `${metrics.specialDisposalPercentage.toFixed(1)}% trash or review needed`,
      icon: AlertTriangle,
      iconBadge: 'bg-amber-50 text-amber-600 border-amber-200/70',
    },
    {
      title: 'Mean Confidence',
      value: loading ? '—' : formatPct(metrics.averageConfidence),
      subtext: 'Dual-model ensemble certainty',
      icon: Percent,
      iconBadge: 'bg-slate-100 text-slate-700 border-slate-200',
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {cards.map((card) => {
        const Icon = card.icon
        return (
          <div
            key={card.title}
            className="stat-card hover:shadow-eco transition-shadow duration-200"
          >
            <div className="flex items-center justify-between gap-3 mb-2">
              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                {card.title}
              </span>
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-xl border shrink-0 ${card.iconBadge}`}
              >
                <Icon className="h-4 w-4" />
              </div>
            </div>
            <div className="text-2xl sm:text-3xl font-heading font-extrabold text-slate-900 tracking-tight">
              {card.value}
            </div>
            <p className="text-xs text-slate-500 mt-1 font-medium truncate">
              {card.subtext}
            </p>
          </div>
        )
      })}
    </div>
  )
}
