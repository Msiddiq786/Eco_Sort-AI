import type { HistoryItem, WasteCategory } from '@/types'
import { capitalize, formatPct, CATEGORY_COLORS, WASTE_CLASSES } from '@/lib'
import { CATEGORY_DETAILS } from '@/components/learning/CategoryInfoCard'
import { X, Layers, Sparkles } from 'lucide-react'

interface PredictionDetailModalProps {
  item: HistoryItem | null
  onClose: () => void
}

export default function PredictionDetailModal({ item, onClose }: PredictionDetailModalProps) {
  if (!item) return null

  const color = CATEGORY_COLORS[item.predicted_category] ?? '#16A34A'
  const details = CATEGORY_DETAILS[item.predicted_category as WasteCategory]
  const probs = item.class_probabilities || {}
  const dateStr = new Date(item.created_at).toLocaleString()

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl border border-slate-100 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div
              className="flex h-10 w-10 items-center justify-center rounded-xl text-white font-bold text-sm shadow-sm"
              style={{ background: color }}
            >
              {capitalize(item.predicted_category).charAt(0)}
            </div>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Prediction #{item.id} Details
              </span>
              <h3 className="font-heading font-bold text-lg text-slate-900 leading-tight">
                {capitalize(item.predicted_category)}
              </h3>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-4 space-y-4 text-xs sm:text-sm">
          {/* Metadata Grid */}
          <div className="grid grid-cols-2 gap-3 p-3 rounded-xl bg-slate-50 border border-slate-200/60 text-xs">
            <div>
              <span className="text-muted-foreground block text-[11px]">Original Filename:</span>
              <span className="font-mono font-medium text-slate-800 break-all">{item.filename}</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Classification Date:</span>
              <span className="font-medium text-slate-800">{dateStr}</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Ensemble Confidence:</span>
              <span className="font-bold text-emerald-700">{formatPct(item.confidence)}</span>
            </div>
            <div>
              <span className="text-muted-foreground block text-[11px]">Model Agreement:</span>
              <span className={`font-semibold ${item.vit_prediction === item.clip_prediction ? 'text-emerald-700' : 'text-amber-700'}`}>
                {item.vit_prediction === item.clip_prediction ? 'Unanimous (100%)' : 'Split Decision'}
              </span>
            </div>
          </div>

          {/* Dual-Model Votes */}
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-slate-200 p-3 bg-white">
              <span className="text-[11px] font-semibold text-muted-foreground block">ViT Model (70% weight)</span>
              <p className="text-sm font-bold text-slate-800 mt-0.5">
                {capitalize(item.vit_prediction)}
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 p-3 bg-white">
              <span className="text-[11px] font-semibold text-muted-foreground block">CLIP Foundation (30% weight)</span>
              <p className="text-sm font-bold text-slate-800 mt-0.5">
                {capitalize(item.clip_prediction)}
              </p>
            </div>
          </div>

          {/* Class Probability Distribution */}
          <div>
            <h4 className="font-semibold text-xs text-slate-700 mb-2 flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5 text-emerald-600" /> Class Probability Breakdown
            </h4>
            <div className="space-y-1.5">
              {WASTE_CLASSES.map(cls => {
                const prob = Number(probs[cls] ?? 0)
                const isWinner = cls === item.predicted_category
                return (
                  <div key={cls} className="space-y-0.5">
                    <div className="flex justify-between text-xs">
                      <span className={`capitalize ${isWinner ? 'font-bold text-emerald-950' : 'text-slate-600'}`}>
                        {cls} {isWinner && '★'}
                      </span>
                      <span className={`font-mono text-[11px] ${isWinner ? 'font-bold text-emerald-700' : 'text-muted-foreground'}`}>
                        {formatPct(prob)}
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.max(2, prob * 100)}%`,
                          background: isWinner ? color : '#CBD5E1',
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Recycling Recommendation */}
          {details && (
            <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-3.5 space-y-1 text-xs text-emerald-950">
              <span className="font-semibold flex items-center gap-1 text-emerald-900">
                <Sparkles className="h-3.5 w-3.5 text-emerald-700" /> Disposal Recommendation:
              </span>
              <p className="text-[11px] text-emerald-800 leading-relaxed">
                Deposit in <strong>{details.binColor}</strong>. {details.preparationTip}
              </p>
            </div>
          )}

          <div className="pt-2 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold transition-colors"
            >
              Close Details
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
