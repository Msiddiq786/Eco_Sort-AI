import { useEffect, useMemo, useRef, useState } from 'react'
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
  X,
} from 'lucide-react'

interface Props {
  imageBase64: string
  imageWidth: number
  imageHeight: number
  objects: DetectedObject[]
  selectedObjectId: number | null
  onSelectObject: (id: number | null) => void
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
  const [hoveredObjectId, setHoveredObjectId] = useState<number | null>(null)
  const [viewportWidth, setViewportWidth] = useState<number>(800)
  const viewportRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const updateWidth = () => {
      const innerW = Math.max(240, el.clientWidth - 16)
      setViewportWidth(innerW)
    }
    updateWidth()
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(updateWidth)
      ro.observe(el)
      return () => ro.disconnect()
    }
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  const safeImgW = Math.max(1, imageWidth || 1)
  const safeImgH = Math.max(1, imageHeight || 1)
  const aspectRatio = safeImgH / safeImgW

  // Base width fits container naturally up to actual pixel width (or full container width)
  // Layout-driven zoom scales exact canvas pixel size cleanly so overflow scrollbars & boxes stay aligned!
  const baseWidth = Math.min(viewportWidth, Math.max(viewportWidth, safeImgW))
  const displayWidth = Math.round(baseWidth * zoom)
  const displayHeight = Math.round(displayWidth * aspectRatio)

  // Filter objects based on materialFilter
  const filteredObjects = useMemo(
    () =>
      objects.filter((obj) => {
        if (materialFilter === 'all') return true
        return obj.material_category.toLowerCase() === materialFilter.toLowerCase()
      }),
    [objects, materialFilter]
  )

  // Sort larger boxes earlier and smaller boxes later so nested/smaller objects remain clickable on top
  const orderedObjects = useMemo(() => {
    return [...filteredObjects].sort((a, b) => {
      const areaA = (a.bbox.x2 - a.bbox.x1) * (a.bbox.y2 - a.bbox.y1)
      const areaB = (b.bbox.x2 - b.bbox.x1) * (b.bbox.y2 - b.bbox.y1)
      return areaB - areaA
    })
  }, [filteredObjects])

  const availableMaterials = Array.from(
    new Set(objects.map((o) => o.material_category.toLowerCase()))
  )

  const hasSelection = selectedObjectId !== null && filteredObjects.some((o) => o.object_id === selectedObjectId)

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm space-y-4">
      {/* ── Toolbar Controls ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100 text-xs">
        <div className="flex flex-wrap items-center gap-2">
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
              onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.25).toFixed(2))))}
              className="p-1 rounded hover:bg-white text-slate-600 hover:text-slate-900 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="px-1 text-[11px] font-mono text-slate-500 font-semibold min-w-[40px] text-center">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(2.5, Number((z + 0.25).toFixed(2))))}
              className="p-1 rounded hover:bg-white text-slate-600 hover:text-slate-900 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
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

          {hasSelection && (
            <button
              type="button"
              onClick={() => onSelectObject(null)}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
            >
              <X className="h-3 w-3" />
              <span>Clear Highlight</span>
            </button>
          )}
        </div>

        {/* Material Filter */}
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

      {/* ── Interactive Alignment-Locked Viewport ───────────────────────────── */}
      <div
        ref={viewportRef}
        className="w-full max-h-[580px] overflow-auto rounded-xl border border-slate-200 bg-slate-100/70 p-2 select-none"
        onClick={(e) => {
          if (e.target === e.currentTarget) {
            onSelectObject(null)
          }
        }}
      >
        <div
          className="relative mx-auto rounded-lg overflow-hidden shadow-sm bg-white"
          style={{
            width: `${displayWidth}px`,
            height: `${displayHeight}px`,
          }}
        >
          {/* Base Image locked strictly to exact displayWidth x displayHeight */}
          <img
            src={imageBase64}
            alt="Multi-object waste detection"
            width={displayWidth}
            height={displayHeight}
            onClick={() => onSelectObject(null)}
            className="w-full h-full block object-fill select-none"
            draggable={false}
          />

          {/* Interactive Bounding Box Overlays */}
          {showBoxes &&
            orderedObjects.map((obj, orderIdx) => {
              const isSelected = selectedObjectId === obj.object_id
              const isHovered = hoveredObjectId === obj.object_id
              const dimmed = hasSelection && !isSelected && !isHovered
              const catColor = CATEGORY_COLORS[obj.material_category.toLowerCase()] ?? '#16A34A'

              const nx1 = obj.bbox.nx1 ?? obj.bbox.x1 / safeImgW
              const ny1 = obj.bbox.ny1 ?? obj.bbox.y1 / safeImgH
              const nx2 = obj.bbox.nx2 ?? obj.bbox.x2 / safeImgW
              const ny2 = obj.bbox.ny2 ?? obj.bbox.y2 / safeImgH

              const leftPct = Math.max(0, Math.min(100, nx1 * 100))
              const topPct = Math.max(0, Math.min(100, ny1 * 100))
              const widthPct = Math.max(0.5, Math.min(100 - leftPct, (nx2 - nx1) * 100))
              const heightPct = Math.max(0.5, Math.min(100 - topPct, (ny2 - ny1) * 100))

              // Keep label inside box if touching top edge (<7%) or aligned right if near right edge (>72%)
              const placeLabelInsideTop = topPct < 7.0
              const alignLabelRight = leftPct > 72.0

              return (
                <div
                  key={obj.object_id}
                  onClick={(e) => {
                    e.stopPropagation()
                    onSelectObject(obj.object_id)
                  }}
                  onMouseEnter={() => setHoveredObjectId(obj.object_id)}
                  onMouseLeave={() => setHoveredObjectId((cur) => (cur === obj.object_id ? null : cur))}
                  className={`absolute cursor-pointer transition-opacity duration-150 ${
                    isSelected
                      ? 'ring-3 ring-white shadow-lg'
                      : isHovered
                        ? 'brightness-110'
                        : ''
                  }`}
                  style={{
                    left: `${leftPct}%`,
                    top: `${topPct}%`,
                    width: `${widthPct}%`,
                    height: `${heightPct}%`,
                    border: `${isSelected ? 3 : 2}px solid ${catColor}`,
                    backgroundColor: isSelected
                      ? `${catColor}24`
                      : dimmed
                        ? 'transparent'
                        : `${catColor}10`,
                    opacity: dimmed ? 0.35 : 1,
                    zIndex: isSelected ? 40 : isHovered ? 35 : 10 + orderIdx,
                  }}
                  title={`Object #${obj.object_id}: ${capitalize(obj.material_category)} (${(obj.calibrated_confidence * 100).toFixed(0)}%) — Click to inspect`}
                >
                  {showLabels && (!dimmed || isSelected || isHovered) && (
                    <div
                      className={`absolute ${
                        placeLabelInsideTop ? 'top-1' : '-top-6'
                      } ${alignLabelRight ? 'right-0' : 'left-0'} px-1.5 py-0.5 rounded text-[10px] font-bold text-white shadow-md flex items-center gap-1 whitespace-nowrap pointer-events-none`}
                      style={{
                        backgroundColor: catColor,
                        zIndex: isSelected ? 45 : 30,
                      }}
                    >
                      <span>#{obj.object_id}</span>
                      <span>{capitalize(obj.material_category)}</span>
                      <span className="opacity-90 font-mono text-[9px]">
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
        Click any bounding box or card to highlight and inspect independent Detection, ViT, CLIP, and Ensemble scores. Click empty space to view all boxes.
      </p>
    </div>
  )
}
