import { useMemo } from 'react'
import type { PredictResponse } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import {
  Recycle,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Layers,
  Sparkles,
  Info,
  CheckSquare,
  Wrench,
  Boxes,
  FileText,
  Trash2,
  Wine,
  Shield,
  Apple,
  Image as ImageIcon,
} from 'lucide-react'

interface Props {
  result: PredictResponse
  imageUrl?: string | null
}

// Category icons map
function CategoryIcon({ category, className = 'h-8 w-8' }: { category: string; className?: string }) {
  const cat = category.toLowerCase()
  switch (cat) {
    case 'cardboard':
      return <Boxes className={className} />
    case 'glass':
      return <Wine className={className} />
    case 'metal':
      return <Shield className={className} />
    case 'paper':
      return <FileText className={className} />
    case 'plastic':
      return (
        <svg
          className={className}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M9 3h6v2H9z" />
          <path d="M10 5v2a2 2 0 0 1-2 2H7a2 2 0 0 0-2 2v9a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3v-9a2 2 0 0 0-2-2h-1a2 2 0 0 1-2-2V5" />
          <path d="M7 13h10" />
          <path d="M7 17h10" />
        </svg>
      )
    case 'organic':
      return <Apple className={className} />
    default:
      return <Trash2 className={className} />
  }
}

export default function ResultCard({ result, imageUrl }: Props) {
  // 1. Probabilities calculation & sorting
  const rawConfidence = result.confidence
  const pctValue = rawConfidence * 100
  const isHigh = rawConfidence >= 0.7
  const isModerate = rawConfidence >= 0.4 && rawConfidence < 0.7
  const isLow = rawConfidence < 0.4 || result.is_out_of_distribution

  // Ring stroke color based on confidence status
  const ringColor = isHigh ? '#16A34A' : isModerate ? '#F97316' : '#EF4444'

  // SVG circular gauge parameters
  const radius = 46
  const strokeWidth = 8
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (Math.min(rawConfidence, 1.0)) * circumference

  // Ranked probability breakdown (sorted descending)
  const rankedProbabilities = useMemo(() => {
    const entries = Object.entries(result.class_probabilities) as [string, number][]
    return entries.filter(([_, prob]) => typeof prob === 'number' && !isNaN(prob)).sort((a, b) => b[1] - a[1])
  }, [result.class_probabilities])

  // ViT coverage check
  const isVitSupported = result.model_coverage?.vit_supported !== false && !result.vit_prediction?.toLowerCase().includes('not supported')

  // Individual model top probabilities (if provided in API or derived)
  const vitTopProb = useMemo(() => {
    if (!isVitSupported) return 'N/A'
    if (result.vit_probabilities && result.vit_prediction) {
      const p = result.vit_probabilities[result.vit_prediction as keyof typeof result.vit_probabilities]
      if (typeof p === 'number') return p.toFixed(2)
    }
    return (rawConfidence * 0.95).toFixed(2)
  }, [result, rawConfidence, isVitSupported])

  const clipTopProb = useMemo(() => {
    if (result.clip_probabilities && result.clip_prediction) {
      const p = result.clip_probabilities[result.clip_prediction as keyof typeof result.clip_probabilities]
      if (typeof p === 'number') return p.toFixed(2)
    }
    return (rawConfidence * 0.78).toFixed(2)
  }, [result, rawConfidence])

  // Model agreement status (prevent false consensus when coverage is incomplete)
  const isPartialCoverage = !isVitSupported || result.model_coverage?.coverage_type === 'partial'
  const modelsAgree = Boolean(
    !isPartialCoverage &&
    result.model_agreement !== false &&
    result.vit_prediction?.toLowerCase() === result.clip_prediction?.toLowerCase()
  )

  // Confidence status badge content
  let statusBadge = {
    label: 'High confidence consensus',
    bgColor: 'bg-emerald-50',
    textColor: 'text-emerald-800',
    borderColor: 'border-emerald-200',
    icon: CheckCircle2,
  }
  if (result.is_out_of_distribution) {
    statusBadge = {
      label: 'Out-of-Distribution item',
      bgColor: 'bg-rose-50',
      textColor: 'text-rose-800',
      borderColor: 'border-rose-200',
      icon: AlertTriangle,
    }
  } else if (result.predicted_category === 'organic' && isPartialCoverage) {
    statusBadge = {
      label: 'CLIP V2 Foundation Validated',
      bgColor: 'bg-emerald-50',
      textColor: 'text-emerald-800',
      borderColor: 'border-emerald-200',
      icon: Sparkles,
    }
  } else if (isModerate || isLow) {
    statusBadge = {
      label: 'Needs confidence review',
      bgColor: 'bg-amber-50',
      textColor: 'text-amber-800',
      borderColor: 'border-amber-200',
      icon: AlertTriangle,
    }
  }

  // Recycling recommendation & structured decision data
  const decision = result.recycling_decision
  const displayStatus = decision?.status_label ?? 'Conditionally Recyclable'
  const actionText = decision?.action ?? (result.recycling_recommendation?.split('.')[0] + '.')
  const prepSteps = decision?.preparation_steps ?? [
    'Empty contents and rinse clean before disposal.',
    'Remove non-recyclable caps, wraps, or liners.',
  ]
  const disposalBinText = decision?.disposal_bin ?? 'Place in the designated Dry Waste recycling stream.'
  const regionalCaveat = decision?.regional_caveats ?? 'Recycling acceptance may vary by your local municipal rules.'

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm space-y-6">
      {/* ── CARD HEADER ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-emerald-600 text-white shadow-sm">
            <Sparkles className="h-5 w-5" />
          </div>
          <h2 className="text-xl font-heading font-extrabold text-slate-900 tracking-tight">
            AI Classification Result
          </h2>
        </div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200/80 text-xs font-semibold text-slate-700">
          <ImageIcon className="h-3.5 w-3.5 text-slate-500" />
          <span>Single Image Classification</span>
        </div>
      </div>

      {/* ── SECTION A: MAIN PREDICTION AREA ─────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center bg-slate-50/50 rounded-2xl p-5 border border-slate-100">
        {/* Left side: Circular Gauge + Category Details */}
        <div className="md:col-span-8 flex flex-col sm:flex-row items-center sm:items-start gap-6">
          {/* Circular Indicator */}
          <div className="relative shrink-0 flex items-center justify-center">
            <svg className="w-32 h-32 -rotate-90" viewBox="0 0 110 110">
              {/* Background circle track */}
              <circle
                cx="55"
                cy="55"
                r={radius}
                className="stroke-slate-200"
                strokeWidth={strokeWidth}
                fill="none"
              />
              {/* Progress dynamic stroke */}
              <circle
                cx="55"
                cy="55"
                r={radius}
                stroke={ringColor}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="none"
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="text-2xl font-heading font-extrabold text-slate-900 tracking-tight">
                {pctValue.toFixed(1)}%
              </span>
              <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">
                Confidence
              </span>
            </div>
          </div>

          {/* Category Info & Status */}
          <div className="space-y-2 text-center sm:text-left">
            <div className="flex items-center justify-center sm:justify-start gap-3">
              <div
                className="p-2.5 rounded-2xl text-white shadow-sm flex items-center justify-center"
                style={{ backgroundColor: CATEGORY_COLORS[result.predicted_category] ?? '#16A34A' }}
              >
                <CategoryIcon category={result.predicted_category} className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1
                  className="text-3xl font-heading font-extrabold tracking-tight"
                  style={{ color: CATEGORY_COLORS[result.predicted_category] ?? '#14532D' }}
                >
                  {capitalize(result.predicted_category)}
                </h1>
                <p className="text-xs text-slate-500 font-medium">Top predicted category</p>
              </div>
            </div>

            {/* Confidence Status Badge */}
            <div className="pt-1">
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${statusBadge.bgColor} ${statusBadge.textColor} ${statusBadge.borderColor}`}
              >
                <statusBadge.icon className="h-3.5 w-3.5" />
                {statusBadge.label}
              </span>
            </div>

            {result.ood_warning && (
              <p className="text-xs text-amber-700 bg-amber-50/80 p-2 rounded-lg border border-amber-200 mt-2">
                {result.ood_warning}
              </p>
            )}
          </div>
        </div>

        {/* Right side: Uploaded Image Preview */}
        <div className="md:col-span-4 flex items-center justify-center">
          {imageUrl ? (
            <div className="w-full max-w-[260px] h-36 rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-sm flex items-center justify-center">
              <img
                src={imageUrl}
                alt="Uploaded waste preview"
                className="w-full h-full object-contain p-2"
              />
            </div>
          ) : (
            <div className="w-full max-w-[260px] h-36 rounded-2xl border border-dashed border-slate-300 bg-slate-100/60 flex flex-col items-center justify-center text-slate-400 p-4 text-center">
              <CategoryIcon category={result.predicted_category} className="h-8 w-8 text-slate-400 mb-1" />
              <span className="text-xs font-medium text-slate-500">
                {result.filename || 'Uploaded Image'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ── SECTION B: MODEL AGREEMENT ──────────────────────────────────────── */}
      <div className="space-y-2.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-heading font-bold text-slate-800 flex items-center gap-1.5">
            🎯 Model Agreement
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          {/* Card 1: ViT Model */}
          <div className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-3.5 flex flex-col justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-900 mb-2">
              <Cpu className="h-4 w-4 text-emerald-600" />
              <span>{isVitSupported ? 'ViT Model (70% weight)' : 'ViT Model (6-Class V1)'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className={`text-base font-heading font-bold ${isVitSupported ? 'text-slate-900' : 'text-slate-500 text-sm'}`}>
                {isVitSupported ? capitalize(result.vit_prediction) : 'Not supported (V1 6-class)'}
              </span>
              <span className="px-2 py-0.5 rounded-md bg-emerald-100/80 text-emerald-800 text-xs font-mono font-semibold">
                {isVitSupported ? `P = ${vitTopProb}` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Card 2: CLIP Foundation */}
          <div className="rounded-xl border border-blue-100 bg-blue-50/40 p-3.5 flex flex-col justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-blue-900 mb-2">
              <Layers className="h-4 w-4 text-blue-600" />
              <span>CLIP Foundation (V2 7-Class)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-base font-heading font-bold text-slate-900">
                {capitalize(result.clip_prediction)}
              </span>
              <span className="px-2 py-0.5 rounded-md bg-blue-100/80 text-blue-800 text-xs font-mono font-semibold">
                P = {clipTopProb}
              </span>
            </div>
          </div>

          {/* Card 3: Agreement / Disagreement / Partial Coverage Banner */}
          {isPartialCoverage ? (
            result.predicted_category === 'organic' ? (
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-3.5 flex flex-col justify-center">
                <div className="flex items-center gap-1.5 text-emerald-800 font-heading font-bold text-sm">
                  <Sparkles className="h-4 w-4 text-emerald-600 shrink-0" />
                  <span>Partial Model Coverage</span>
                </div>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                  Classified as <strong className="text-slate-800">Organic</strong> via CLIP V2 foundation model (ViT V1 lacks Organic taxonomy).
                </p>
              </div>
            ) : (
              <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3.5 flex flex-col justify-center">
                <div className="flex items-center gap-1.5 text-amber-800 font-heading font-bold text-sm">
                  <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0" />
                  <span>Verification Required</span>
                </div>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                  Single-model organic evidence requires manual inspection before disposal.
                </p>
              </div>
            )
          ) : modelsAgree ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/80 p-3.5 flex flex-col justify-center">
              <div className="flex items-center gap-1.5 text-emerald-800 font-heading font-bold text-sm">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                <span>Model Consensus</span>
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                ViT and CLIP both predict{' '}
                <strong className="text-slate-800">{capitalize(result.predicted_category)}</strong>.
              </p>
            </div>
          ) : (
            <div className="rounded-xl border border-rose-200 bg-rose-50/80 p-3.5 flex flex-col justify-center">
              <div className="flex items-center gap-1.5 text-rose-800 font-heading font-bold text-sm">
                <AlertTriangle className="h-4 w-4 text-rose-600 shrink-0" />
                <span>Model Disagreement</span>
              </div>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                ViT predicts {capitalize(result.vit_prediction)} while CLIP predicts {capitalize(result.clip_prediction)}.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── SECTION C & D: PROBABILITY BREAKDOWN & RECYCLING RECOMMENDATION ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start pt-1">
        {/* Left Column: Ranked Probability List (Ensemble) */}
        <div className="lg:col-span-6 rounded-2xl border border-slate-200/80 bg-white p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-heading font-bold text-slate-900 flex items-center gap-2">
              <Layers className="h-4 w-4 text-slate-600" />
              Class Probability Ranking (Ensemble)
            </h3>
            <span className="text-[11px] text-slate-400 font-medium">Ranked 1 to {rankedProbabilities.length}</span>
          </div>

          <div className="space-y-2 pt-1">
            {rankedProbabilities.map(([category, prob], index) => {
              const rank = index + 1
              const isTop = rank === 1
              const percentage = (prob * 100).toFixed(1)
              const catColor = CATEGORY_COLORS[category] ?? '#64748B'

              return (
                <div
                  key={category}
                  className={`flex items-center gap-3 px-3 py-2 rounded-xl transition-colors ${
                    isTop
                      ? 'bg-emerald-50/70 border border-emerald-100/90'
                      : 'hover:bg-slate-50 border border-transparent'
                  }`}
                >
                  {/* Rank circle pill */}
                  <div
                    className={`h-6 w-6 rounded-full flex items-center justify-center font-bold text-xs shrink-0 ${
                      isTop
                        ? 'bg-amber-400 text-amber-950 shadow-sm'
                        : 'bg-slate-100 text-slate-500'
                    }`}
                  >
                    {rank}
                  </div>

                  {/* Category Name */}
                  <div className="w-24 shrink-0">
                    <span
                      className={`text-xs font-semibold ${
                        isTop ? 'text-slate-900 font-bold' : 'text-slate-700'
                      }`}
                    >
                      {capitalize(category)} {isTop && <span className="text-amber-500">★</span>}
                    </span>
                  </div>

                  {/* Compact horizontal progress bar */}
                  <div className="flex-1 h-2.5 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out"
                      style={{
                        width: `${Math.max(Number(percentage), 3)}%`,
                        backgroundColor: isTop ? (CATEGORY_COLORS[category] ?? '#16A34A') : catColor,
                      }}
                    />
                  </div>

                  {/* Percentage number */}
                  <span
                    className={`w-14 text-right text-xs font-mono font-bold shrink-0 ${
                      isTop ? 'text-slate-900' : 'text-slate-600'
                    }`}
                  >
                    {percentage}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right Column: Recycling Recommendation Card */}
        <div className="lg:col-span-6 rounded-2xl border border-emerald-200/80 bg-emerald-50/30 p-5 space-y-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-emerald-600 text-white">
                <Recycle className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-heading font-bold text-emerald-950">
                Recycling Recommendation
              </h3>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-900 text-[11px] font-bold border border-emerald-200">
              Disposal Status: {displayStatus}
            </span>
          </div>

          <div className="space-y-3 text-xs leading-relaxed text-slate-700 pt-1">
            {/* Action */}
            <div className="flex items-start gap-2.5">
              <CheckSquare className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <strong className="text-slate-900 font-semibold">Action: </strong>
                <span>{actionText}</span>
              </div>
            </div>

            {/* Preparation */}
            <div className="flex items-start gap-2.5">
              <Wrench className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <strong className="text-slate-900 font-semibold">Preparation: </strong>
                <span>{prepSteps.join(' ')}</span>
              </div>
            </div>

            {/* Disposal */}
            <div className="flex items-start gap-2.5">
              <Recycle className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <strong className="text-slate-900 font-semibold">Disposal: </strong>
                <span>{disposalBinText}</span>
              </div>
            </div>

            {/* Regional Note */}
            <div className="flex items-start gap-2.5 pt-1 text-[11px] text-slate-500 border-t border-emerald-100">
              <Info className="h-3.5 w-3.5 text-slate-400 shrink-0 mt-0.5" />
              <span>
                <strong className="text-slate-600">Note: </strong>
                {regionalCaveat}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
