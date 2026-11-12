import type { DetectedObject } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import { Cpu, Layers, Recycle, ChevronRight, CheckCircle2, AlertCircle, Crop } from 'lucide-react'

interface Props {
  object: DetectedObject
  isSelected: boolean
  onSelect: () => void
}

export default function ObjectCard({ object, isSelected, onSelect }: Props) {
  const catColor = CATEGORY_COLORS[object.material_category.toLowerCase()] ?? '#16A34A'
  const isAgree = object.model_agreement ?? (object.vit_prediction.toLowerCase() === object.clip_prediction.toLowerCase())

  return (
    <div
      onClick={onSelect}
      className={`rounded-2xl border p-4 cursor-pointer transition-all duration-150 flex flex-col justify-between space-y-3 ${
        isSelected
          ? 'border-emerald-500 bg-emerald-50/40 shadow-md ring-2 ring-emerald-500/20'
          : 'border-slate-200/80 bg-white hover:border-slate-300 hover:shadow-xs'
      }`}
    >
      {/* Top Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div
            className="h-7 w-7 rounded-xl flex items-center justify-center font-bold text-xs text-white shrink-0 shadow-xs"
            style={{ backgroundColor: catColor }}
          >
            #{object.object_id}
          </div>
          <div>
            <h4 className="text-sm font-heading font-extrabold text-slate-900 leading-tight">
              {object.detection_label || `${capitalize(object.material_category)} item`}
            </h4>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: catColor }}
              />
              <span className="text-xs font-semibold" style={{ color: catColor }}>
                {capitalize(object.material_category)}
              </span>
              {object.model_coverage?.vit_supported === false && (
                <span className="px-1.5 py-0.2 rounded text-[9px] font-semibold bg-amber-100 text-amber-800">
                  CLIP V2
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 text-[10px] font-mono font-bold">
            Cls {(object.calibrated_confidence * 100).toFixed(0)}%
          </span>
          <span className="text-[10px] font-mono text-slate-400">
            Det {(object.detection_confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Model Predictions & Agreement */}
      <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-slate-100">
        <div className="flex items-center gap-1.5 text-slate-600 truncate">
          <Cpu className="h-3 w-3 text-emerald-600 shrink-0" />
          <span className="truncate">ViT: <strong>{capitalize(object.vit_prediction)}</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-600 truncate">
          <Layers className="h-3 w-3 text-blue-600 shrink-0" />
          <span className="truncate">CLIP: <strong>{capitalize(object.clip_prediction)}</strong></span>
        </div>
      </div>

      {/* Crop and Agreement Indicators */}
      <div className="flex items-center justify-between text-[10px] text-slate-500 pt-0.5">
        <div className="flex items-center gap-1">
          {isAgree ? (
            <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
              <CheckCircle2 className="h-3 w-3 text-emerald-600" />
              <span>Consensus</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-amber-700 font-medium">
              <AlertCircle className="h-3 w-3 text-amber-600" />
              <span>Disagreement</span>
            </span>
          )}
        </div>

        {object.crop_width && object.crop_height ? (
          <span className="inline-flex items-center gap-1 font-mono text-slate-400">
            <Crop className="h-2.5 w-2.5" />
            {object.crop_width}x{object.crop_height}
          </span>
        ) : null}
      </div>

      {/* Disposal recommendation footer */}
      <div className="flex items-center justify-between pt-1 border-t border-slate-100 text-xs">
        <div className="flex items-center gap-1.5 text-emerald-800 font-medium">
          <Recycle className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
          <span className="text-[11px] truncate max-w-[180px]">
            {object.recycling_decision?.status_label || object.recycling_status}
          </span>
        </div>
        <ChevronRight className={`h-4 w-4 text-slate-400 transition-transform ${isSelected ? 'rotate-90 text-emerald-600' : ''}`} />
      </div>
    </div>
  )
}
