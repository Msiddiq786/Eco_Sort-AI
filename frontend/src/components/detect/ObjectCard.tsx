import type { DetectedObject } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import { Cpu, Layers, Recycle, ChevronRight } from 'lucide-react'

interface Props {
  object: DetectedObject
  isSelected: boolean
  onSelect: () => void
}

export default function ObjectCard({ object, isSelected, onSelect }: Props) {
  const catColor = CATEGORY_COLORS[object.material_category.toLowerCase()] ?? '#16A34A'

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
            </div>
          </div>
        </div>

        <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 text-[10px] font-mono font-bold shrink-0">
          {(object.calibrated_confidence * 100).toFixed(0)}% conf
        </span>
      </div>

      {/* Model Predictions */}
      <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-slate-100">
        <div className="flex items-center gap-1.5 text-slate-600">
          <Cpu className="h-3 w-3 text-emerald-600 shrink-0" />
          <span className="truncate">ViT: <strong>{capitalize(object.vit_prediction)}</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-600">
          <Layers className="h-3 w-3 text-blue-600 shrink-0" />
          <span className="truncate">CLIP: <strong>{capitalize(object.clip_prediction)}</strong></span>
        </div>
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
