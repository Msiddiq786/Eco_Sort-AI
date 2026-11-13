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
      iconColor: 'bg-emerald-600 text-white',
      accent: 'border-l-4 border-l-emerald-600',
    },
    {
      title: 'Most Detected Category',
      value: loading
        ? '—'
        : metrics.mostDetectedCategory
        ? capitalize(metrics.mostDetectedCategory.category)
        : 'None',
      subtext: loading
        ? ''
        : metrics.mostDetectedCategory
        ? `${metrics.mostDetectedCategory.count} items (${metrics.mostDetectedCategory.percentage.toFixed(1)}% of total)`
        : 'No records in range',
      icon: TrendingUp,
      iconColor: 'bg-blue-600 text-white',
      accent: 'border-l-4 border-l-blue-600',
    },
    {
      title: 'Recyclable Items',
      value: loading ? '—' : metrics.recyclableCount.toLocaleString(),
      subtext: loading
        ? ''
        : `${metrics.recyclablePercentage.toFixed(1)}% diverted from landfill`,
      icon: Recycle,
      iconColor: 'bg-teal-600 text-white',
      accent: 'border-l-4 border-l-teal-600',
    },
    {
      title: 'Compostable Items',
      value: loading ? '—' : metrics.compostableCount.toLocaleString(),
      subtext: loading
        ? ''
        : `${metrics.compostablePercentage.toFixed(1)}% organic wet waste`,
      icon: Apple,
      iconColor: 'bg-lime-600 text-white',
      accent: 'border-l-4 border-l-lime-600',
    },
    {
      title: 'Special Disposal / Residual',
      value: loading ? '—' : metrics.specialDisposalCount.toLocaleString(),
      subtext: loading
        ? ''
        : `${metrics.specialDisposalPercentage.toFixed(1)}% trash or verification needed`,
      icon: AlertTriangle,
      iconColor: 'bg-amber-600 text-white',
      accent: 'border-l-4 border-l-amber-600',
    },
    {
      title: 'Average Model Confidence',
      value: loading ? '—' : formatPct(metrics.averageConfidence),
      subtext: 'Dual-model ensemble mean confidence',
      icon: Percent,
      iconColor: 'bg-purple-600 text-white',
      accent: 'border-l-4 border-l-purple-600',
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon
        return (
          <div
            key={idx}
            className={`rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm hover:shadow-md transition-shadow duration-200 ${card.accent}`}
          >
            <div className="flex items-center justify-between gap-3 mb-2">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                {card.title}
              </span>
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-xl shadow-sm shrink-0 ${card.iconColor}`}
              >
                <Icon className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-1">
              <div className="text-2xl sm:text-3xl font-heading font-extrabold text-slate-900 tracking-tight">
                {card.value}
              </div>
              <p className="text-xs text-slate-500 mt-1 font-medium truncate">
                {card.subtext}
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}
