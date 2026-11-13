import { useMemo } from 'react'
import type { HistoryItem } from '@/types'
import {
  CheckCircle,
  HelpCircle,
  Leaf,
  AlertTriangle,
  RotateCcw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'

interface RecyclingInsightsSectionProps {
  items: HistoryItem[]
  loading?: boolean
}

export default function RecyclingInsightsSection({
  items,
  loading = false,
}: RecyclingInsightsSectionProps) {
  const insights = useMemo(() => {
    if (loading || !items || items.length === 0) {
      return {
        recyclable: 0,
        conditionallyRecyclable: 0,
        compostable: 0,
        nonStandard: 0,
        hazardousOrTrash: 0,
        verificationRequired: 0,
        total: 0,
      }
    }

    let recyclable = 0
    let conditionallyRecyclable = 0
    let compostable = 0
    let nonStandard = 0
    let hazardousOrTrash = 0
    let verificationRequired = 0

    const recyclableCats = new Set(['cardboard', 'glass', 'metal', 'paper', 'plastic'])

    for (const item of items) {
      const cat = (item.predicted_category || '').toLowerCase()
      const vit = (item.vit_prediction || '').toLowerCase()
      const clip = (item.clip_prediction || '').toLowerCase()
      const conf = item.confidence || 0

      // 1. Verification required: Split decision / disagreement OR low confidence (< 0.50)
      const isDisagreement = vit && clip && vit !== clip && !vit.includes('not supported')
      const isCappedDisagreement = conf === 0.68 && isDisagreement
      if (isCappedDisagreement || conf < 0.50) {
        verificationRequired++
      }
      // 2. Compostable: Organic items
      else if (cat === 'organic' || clip === 'organic') {
        compostable++
      }
      // 3. Trash / Hazardous
      else if (cat === 'trash') {
        hazardousOrTrash++
      }
      // 4. Conditionally recyclable: Recyclable category but moderate confidence (0.50 - 0.75)
      else if (recyclableCats.has(cat) && conf < 0.75) {
        conditionallyRecyclable++
      }
      // 5. Standard recyclable: High confidence (>= 0.75) and consensus
      else if (recyclableCats.has(cat)) {
        recyclable++
      }
      // 6. Non-standard fallback
      else {
        nonStandard++
      }
    }

    return {
      recyclable,
      conditionallyRecyclable,
      compostable,
      nonStandard,
      hazardousOrTrash,
      verificationRequired,
      total: items.length,
    }
  }, [items])

  const total = insights.total || 1

  const insightCards = [
    {
      title: 'Widely Recyclable',
      count: insights.recyclable,
      percentage: (insights.recyclable / total) * 100,
      icon: CheckCircle,
      color: 'bg-emerald-600',
      badge: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      description: 'Accepted in standard blue/yellow curbside bins: clean cans, glass bottles, rigid plastic & cardboard.',
      action: 'Keep clean & dry; flatten cardboard boxes.',
    },
    {
      title: 'Conditionally Recyclable',
      count: insights.conditionallyRecyclable,
      percentage: (insights.conditionallyRecyclable / total) * 100,
      icon: RotateCcw,
      color: 'bg-blue-600',
      badge: 'bg-blue-100 text-blue-800 border-blue-200',
      description: 'Recyclable subject to preparation: requires rinsing food residue, separating paper from plastic films.',
      action: 'Thoroughly rinse food containers before depositing.',
    },
    {
      title: 'Compostable Biomass',
      count: insights.compostable,
      percentage: (insights.compostable / total) * 100,
      icon: Leaf,
      color: 'bg-lime-600',
      badge: 'bg-lime-100 text-lime-800 border-lime-200',
      description: 'Fruit peels, vegetable waste, and organic scraps. Diverted to compost or green wet waste carts.',
      action: 'Deposit in green bin; remove plastic stickers.',
    },
    {
      title: 'Non-Standard Recyclable',
      count: insights.nonStandard,
      percentage: (insights.nonStandard / total) * 100,
      icon: Sparkles,
      color: 'bg-sky-600',
      badge: 'bg-sky-100 text-sky-800 border-sky-200',
      description: 'Requires specialized collection (e.g. supermarket soft plastic drop-off, scrap metal depots).',
      action: 'Collect for grocery store plastic film drop-off.',
    },
    {
      title: 'Residual Waste (Landfill / WTE)',
      count: insights.hazardousOrTrash,
      percentage: (insights.hazardousOrTrash / total) * 100,
      icon: AlertTriangle,
      color: 'bg-slate-600',
      badge: 'bg-slate-100 text-slate-800 border-slate-200',
      description: 'Multi-layer composite wrappers, snack bags, and non-recyclable domestic residual refuse.',
      action: 'Discard in dark grey/black residual bin.',
    },
    {
      title: 'Verification Required',
      count: insights.verificationRequired,
      percentage: (insights.verificationRequired / total) * 100,
      icon: HelpCircle,
      color: 'bg-amber-600',
      badge: 'bg-amber-100 text-amber-800 border-amber-200',
      description: 'Ambiguous items with split model decisions or lower confidence scores requiring manual review.',
      action: 'Inspect image preview and confirm material manually.',
    },
  ]

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
      <div>
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-emerald-600" />
          <h3 className="font-heading font-bold text-base text-slate-900">
            Intelligent Recycling Decision Engine Statuses
          </h3>
        </div>
        <p className="text-xs text-slate-500 mt-0.5">
          Real-world classification outcomes partitioned by circular economy disposition protocols
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {insightCards.map((card, idx) => {
          const Icon = card.icon
          return (
            <div
              key={idx}
              className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 hover:bg-slate-50 transition flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded-lg text-white shadow-xs shrink-0 ${card.color}`}
                    >
                      <Icon className="h-3.5 w-3.5" />
                    </div>
                    <span className="font-heading font-bold text-xs text-slate-800">
                      {card.title}
                    </span>
                  </div>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${card.badge}`}
                  >
                    {card.count} ({card.percentage.toFixed(1)}%)
                  </span>
                </div>

                <p className="text-[11px] text-slate-600 leading-relaxed mb-2">
                  {card.description}
                </p>

                {/* Mini progress bar */}
                <div className="w-full bg-slate-200/70 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${card.color}`}
                    style={{ width: `${Math.min(100, Math.max(0, card.percentage))}%` }}
                  />
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200/60 text-[11px] text-slate-500 flex items-center gap-1.5">
                <span className="font-bold text-slate-700">Action:</span>
                <span className="truncate">{card.action}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
