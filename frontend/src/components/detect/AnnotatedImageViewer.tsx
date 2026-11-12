import { useState, useRef } from 'react'
import type { DetectedObject } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Eye,
  EyeOff,
  Tag,
  Filter,
} from 'lucide-react'

interface Props {
  imageBase64: string
  imageWidth: number
  imageHeight: number
  objects: DetectedObject[]
  selectedObjectId: number | null
  onSelectObject: (id: number) => void
  materialFilter: string
  onFilterChange: (filter: string) => void
}

export default function AnnotatedImageViewer({
  imageBase64,
  imageWidth,
  imageHeight,
  objects,
  selectedObjectId,
  onSelectObject,
  materialFilter,
  onFilterChange,
}: Props) {
  const [zoom, setZoom] = useState(1.0)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)

  // Filter objects based on materialFilter
  const filteredObjects = objects.filter((obj) => {
    if (materialFilter === 'all') return true
    return obj.material_category.toLowerCase() === materialFilter.toLowerCase()
  })

  // Unique materials in detections
  const availableMaterials = Array.from(
    new Set(objects.map((o) => o.material_category.toLowerCase()))
  )

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm space-y-4">
      {/* ── Toolbar Controls ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100 text-xs">
        {/* Left: View controls */}
        <div className="flex items-center gap-2">
          {/* Toggle Boxes */}
          <button
            type="button"
            onClick={() => setShowBoxes(!showBoxes)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-colors border ${
              showBoxes
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                : 'bg-slate-50 text-slate-600 border-slate-200'
            }`}
          >
            {showBoxes ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
            <span>Boxes</span>
          </button>

          {/* Toggle Labels */}
          <button
            type="button"
            onClick={() => setShowLabels(!showLabels)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-colors border ${
              showLabels
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                : 'bg-slate-50 text-slate-600 border-slate-200'
            }`}
          >
            <Tag className="h-3.5 w-3.5" />
            <span>Labels</span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 border border-slate-200 rounded-lg p-0.5 bg-slate-50">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
              className="p-1 rounded hover:bg-white text-slate-600 hover:text-slate-900 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
            <span className="px-1 text-[11px] font-mono text-slate-500 font-semibold">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
              className="p-1 rounded hover:bg-white text-slate-600 hover:text-slate-900 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setZoom(1.0)}
              className="p-1 rounded hover:bg-white text-slate-600 hover:text-slate-900 transition-colors border-l border-slate-200 ml-0.5"
              title="Reset Zoom"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Right: Material Filter */}
        <div className="flex items-center gap-2">
          <Filter className="h-3.5 w-3.5 text-slate-400" />
          <span className="text-slate-500 font-medium">Filter Material:</span>
          <select
            value={materialFilter}
            onChange={(e) => onFilterChange(e.target.value)}
            className="px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-700 text-xs font-medium focus:outline-hidden focus:ring-2 focus:ring-emerald-500/20"
          >
            <option value="all">All Materials ({objects.length})</option>
            {availableMaterials.map((mat) => {
              const count = objects.filter((o) => o.material_category.toLowerCase() === mat).length
              return (
                <option key={mat} value={mat}>
                  {capitalize(mat)} ({count})
                </option>
              )
            })}
          </select>
        </div>
      </div>

      {/* ── Interactive Viewport ────────────────────────────────────────────── */}
      <div
        ref={containerRef}
        className="relative w-full max-h-[520px] overflow-auto rounded-xl border border-slate-200 bg-slate-950/5 flex items-center justify-center p-2 select-none"
      >
        <div
          className="relative inline-block transition-transform duration-200 ease-out origin-top-left"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Base Image */}
          <img
            src={imageBase64}
            alt="Multi-object waste detection"
            className="max-w-full h-auto block rounded-lg shadow-sm"
          />

          {/* Interactive Bounding Box Overlays */}
          {showBoxes &&
            filteredObjects.map((obj) => {
              const isSelected = selectedObjectId === obj.object_id
              const catColor = CATEGORY_COLORS[obj.material_category.toLowerCase()] ?? '#16A34A'

              // Percentages relative to image dimensions
              const leftPct = (obj.bbox.x1 / imageWidth) * 100
              const topPct = (obj.bbox.y1 / imageHeight) * 100
              const widthPct = ((obj.bbox.x2 - obj.bbox.x1) / imageWidth) * 100
              const heightPct = ((obj.bbox.y2 - obj.bbox.y1) / imageHeight) * 100

              return (
                <div
                  key={obj.object_id}
                  onClick={() => onSelectObject(obj.object_id)}
                  className={`absolute cursor-pointer transition-all duration-150 ${
                    isSelected
                      ? 'ring-4 ring-white ring-offset-2 ring-offset-slate-900 z-30'
                      : 'hover:brightness-110 z-10'
                  }`}
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                    border: `3px solid ${catColor}`,
                    backgroundColor: isSelected ? `${catColor}25` : `${catColor}10`,
                  }}
                  title={`Object #${obj.object_id}: ${capitalize(obj.material_category)} (Click to inspect)`}
                >
                  {/* Label Pill */}
                  {showLabels && (
                    <div
                      className="absolute -top-7 left-0 px-2 py-0.5 rounded-md text-[11px] font-bold text-white shadow-md flex items-center gap-1 whitespace-nowrap z-20"
                      style={{ backgroundColor: catColor }}
                    >
                      <span>#{obj.object_id}</span>
                      <span>{capitalize(obj.material_category)}</span>
                      <span className="opacity-90 font-mono text-[10px]">
                        {(obj.calibrated_confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
              )
            })}
        </div>
      </div>
      <p className="text-[11px] text-slate-400 text-center">
        Click any bounding box or object card below to inspect individual ViT + CLIP classification and recycling steps.
      </p>
    </div>
  )
}
