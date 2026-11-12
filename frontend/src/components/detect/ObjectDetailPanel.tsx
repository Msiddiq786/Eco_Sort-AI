import type { DetectedObject } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import {
  Recycle,
  AlertTriangle,
  Cpu,
  Layers,
  Info,
  CheckSquare,
  Wrench,
  X,
  Target,
  CheckCircle2,
  AlertCircle,
  Crop,
} from 'lucide-react'

interface Props {
  object: DetectedObject
  onClose: () => void
}

export default function ObjectDetailPanel({ object, onClose }: Props) {
  const cat = object.material_category.toLowerCase()
  const catColor = CATEGORY_COLORS[cat] ?? '#16A34A'
  const pctValue = object.calibrated_confidence * 100

  // SVG circular gauge
  const radius = 38
  const strokeWidth = 7
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - Math.min(object.calibrated_confidence, 1.0) * circumference

  const isHigh = object.calibrated_confidence >= 0.7
  const isModerate = object.calibrated_confidence >= 0.4 && object.calibrated_confidence < 0.7
  const ringColor = isHigh ? '#16A34A' : isModerate ? '#F97316' : '#EF4444'

  const decision = object.recycling_decision
  const isAgree = object.model_agreement ?? (object.vit_prediction.toLowerCase() === object.clip_prediction.toLowerCase())

  return (
    <div className="rounded-2xl border-2 border-emerald-500/40 bg-white p-6 shadow-lg space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-emerald-600 text-white">
            <Target className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Detailed Inspection
              </span>
              <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 font-mono text-[10px] font-bold">
                OBJECT #{object.object_id}
              </span>
            </div>
            <h3 className="text-xl font-heading font-extrabold text-slate-900 leading-tight">
              {object.detection_label || `${capitalize(object.material_category)}`}
            </h3>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-xl border border-slate-200 text-slate-400 hover:text-slate-700 hover:bg-slate-50 transition-colors"
          title="Close details"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Main classification gauge & info */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-center bg-slate-50/70 p-4 rounded-xl border border-slate-100">
        <div className="sm:col-span-4 flex items-center justify-center">
          <div className="relative flex items-center justify-center">
            <svg className="w-28 h-28 -rotate-90" viewBox="0 0 95 95">
              <circle
                cx="47.5"
                cy="47.5"
                r={radius}
                className="stroke-slate-200"
                strokeWidth={strokeWidth}
                fill="none"
              />
              <circle
                cx="47.5"
                cy="47.5"
                r={radius}
                stroke={ringColor}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="none"
                className="transition-all duration-700 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="text-xl font-heading font-extrabold text-slate-900">
                {pctValue.toFixed(1)}%
              </span>
              <span className="text-[10px] font-medium text-slate-500 uppercase">
                Classification
              </span>
            </div>
          </div>
        </div>

        <div className="sm:col-span-8 space-y-1.5">
          <span className="text-xs text-slate-400 font-medium">Final Classification:</span>
          <h2
            className="text-2xl font-heading font-extrabold"
            style={{ color: catColor }}
          >
            {capitalize(object.material_category)}
          </h2>
          <div className="flex flex-wrap gap-2 pt-1 text-xs">
            <span className="px-2.5 py-0.5 rounded-full bg-slate-200/80 text-slate-700 font-medium">
              Spatial Detection: {(object.detection_confidence * 100).toFixed(0)}%
            </span>
            <span className={`px-2.5 py-0.5 rounded-full font-bold ${
              !isAgree || object.recycling_status === 'UNKNOWN_REQUIRES_VERIFICATION'
                ? 'bg-amber-100 text-amber-800 border border-amber-200'
                : 'bg-emerald-100 text-emerald-800'
            }`}>
              {decision?.status_label || (isAgree ? object.recycling_status : 'Inspection Required')}
            </span>
            <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-mono text-[11px]">
              Status: {object.confidence_status}
            </span>
          </div>
        </div>
      </div>

      {/* Model Agreement row */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-slate-700 uppercase tracking-wide text-[11px]">
            Model Ensemble Breakdown
          </span>
          {isAgree ? (
            <span className="inline-flex items-center gap-1 text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-200">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
              <span>Full Consensus</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-amber-800 font-semibold bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
              <AlertCircle className="h-3.5 w-3.5 text-amber-600" />
              <span>Model Disagreement</span>
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-3 flex flex-col justify-between space-y-1">
            <div className="flex items-center justify-between text-xs font-semibold text-emerald-900">
              <div className="flex items-center gap-1.5">
                <Cpu className="h-4 w-4 text-emerald-600" />
                <span>ViT Fine-Tuned (70%):</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-700">
                {object.model_coverage?.vit_supported === false ? 'No native class' : 'Active'}
              </span>
            </div>
            <strong className="text-sm font-heading font-bold text-slate-900">
              {capitalize(object.vit_prediction)}
            </strong>
          </div>

          <div className="rounded-xl border border-blue-100 bg-blue-50/40 p-3 flex flex-col justify-between space-y-1">
            <div className="flex items-center justify-between text-xs font-semibold text-blue-900">
              <div className="flex items-center gap-1.5">
                <Layers className="h-4 w-4 text-blue-600" />
                <span>CLIP Zero-Shot (30%):</span>
              </div>
              <span className="text-[10px] font-mono text-blue-700">
                {object.model_coverage?.clip_supported ? 'Active V2' : 'Active'}
              </span>
            </div>
            <strong className="text-sm font-heading font-bold text-slate-900">
              {capitalize(object.clip_prediction)}
            </strong>
          </div>
        </div>

        {object.model_coverage?.unsupported_reason && (
          <p className="text-[11px] text-amber-700 bg-amber-50/80 p-2 rounded-lg border border-amber-200/80">
            ℹ️ {object.model_coverage.unsupported_reason}
          </p>
        )}
      </div>

      {/* Localization & Crop details */}
      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Crop className="h-4 w-4 text-slate-500" />
          <span className="font-medium text-slate-700">Crop Dimensions:</span>
          <span className="font-mono text-slate-900 font-semibold">
            {object.crop_width && object.crop_height ? `${object.crop_width} × ${object.crop_height} px` : 'Auto-cropped'}
          </span>
        </div>
        <div className="font-mono text-[11px] text-slate-500">
          Bounding Box: [{object.bbox.x1}, {object.bbox.y1}, {object.bbox.x2}, {object.bbox.y2}]
        </div>
      </div>

      {/* Structured recycling recommendation */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/30 p-4 space-y-2.5 text-xs text-slate-700">
        <div className="flex items-center justify-between pb-1 border-b border-emerald-100">
          <div className="flex items-center gap-1.5 font-bold text-emerald-950">
            <Recycle className="h-4 w-4 text-emerald-600" />
            <span>Recycling Decision</span>
          </div>
          <span className="text-[11px] font-semibold text-emerald-800">
            {decision?.disposal_bin || 'Designated Collection Stream'}
          </span>
        </div>

        <div className="space-y-2 pt-1 leading-relaxed">
          <div className="flex items-start gap-2">
            <CheckSquare className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
            <span>
              <strong className="text-slate-900">Action: </strong>
              {object.disposal_recommendation || decision?.action}
            </span>
          </div>

          {object.preparation_steps && object.preparation_steps.length > 0 && (
            <div className="flex items-start gap-2">
              <Wrench className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
              <span>
                <strong className="text-slate-900">Preparation: </strong>
                {object.preparation_steps.join(' ')}
              </span>
            </div>
          )}

          {decision?.safety_info && (
            <div className="flex items-start gap-2 text-rose-700 bg-rose-50 p-2 rounded-lg border border-rose-100">
              <AlertTriangle className="h-3.5 w-3.5 text-rose-600 shrink-0 mt-0.5" />
              <span>
                <strong>Safety Warning: </strong>
                {decision.safety_info}
              </span>
            </div>
          )}

          <div className="flex items-start gap-2 pt-1 text-[11px] text-slate-400">
            <Info className="h-3 w-3 shrink-0 mt-0.5" />
            <span>
              {decision?.regional_caveats || 'Disposal rules and bins vary according to local municipal solid waste guidelines.'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
