import { useState } from 'react'
import { Crosshair, RefreshCw, Layers, Sparkles, Scissors } from 'lucide-react'
import DropZone from '@/components/classify/DropZone'
import AnnotatedImageViewer from '@/components/detect/AnnotatedImageViewer'
import ObjectCard from '@/components/detect/ObjectCard'
import ObjectDetailPanel from '@/components/detect/ObjectDetailPanel'
import RecyclingSummaryCard from '@/components/detect/RecyclingSummaryCard'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
import CategoryBadge from '@/components/ui/CategoryBadge'
import { detectObjects } from '@/services/api'
import { V2_WASTE_CLASSES, type DetectResponse } from '@/types'

export default function DetectPage() {
  const [result, setResult] = useState<DetectResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null)
  const [materialFilter, setMaterialFilter] = useState<string>('all')

  const handleFile = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setSelectedObjectId(null)
    setMaterialFilter('all')

    try {
      const res = await detectObjects(file)
      setResult(res)
      setSelectedObjectId(null)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'An error occurred during multi-object detection.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const selectedObject = result?.objects.find((o) => o.object_id === selectedObjectId)

  const displayedObjects =
    result?.objects.filter((o) =>
      materialFilter === 'all'
        ? true
        : o.material_category.toLowerCase() === materialFilter.toLowerCase()
    ) ?? []

  return (
    <div className="space-y-6 pb-12">
      <PageHeader
        eyebrow="Hybrid Localization + Batched Ensemble"
        eyebrowIcon={Sparkles}
        badge="V3 Beta"
        icon={Crosshair}
        title="Multi-Object Waste Detection"
        subtitle="Isolate, localize, and segregate multiple waste items in a single scene with per-object ViT + CLIP scoring, resin detection, and visual XAI."
        actions={
          result ? (
            <button
              type="button"
              onClick={() => {
                setResult(null)
                setSelectedObjectId(null)
                setMaterialFilter('all')
              }}
              className="btn-secondary text-xs px-3.5 py-2"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Upload Another Scene</span>
            </button>
          ) : null
        }
      />

      {/* Upload Zone (when no result) */}
      {!result && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-8">
            <DropZone onFile={handleFile} loading={loading} />
          </div>

          <aside className="lg:col-span-4 card-eco p-5 space-y-4">
            <h2 className="font-heading font-bold text-sm text-slate-900">
              How Scene Auditing Works
            </h2>
            <ul className="space-y-3 text-xs text-slate-600">
              <li className="flex items-start gap-2.5">
                <span className="p-1.5 rounded-lg bg-emerald-50 text-ecosort-emerald shrink-0">
                  <Crosshair className="h-3.5 w-3.5" />
                </span>
                <span>
                  <strong className="text-slate-800 block">1. Hybrid Object Localization</strong>
                  Combines YOLOv8 trash segmentation, COCO object detection, and residual LAB saliency.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="p-1.5 rounded-lg bg-sky-50 text-sky-600 shrink-0">
                  <Scissors className="h-3.5 w-3.5" />
                </span>
                <span>
                  <strong className="text-slate-800 block">2. Aspect-Preserving Cropping</strong>
                  Extracts padded item crops cleanly to prevent edge-clipping errors.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="p-1.5 rounded-lg bg-amber-50 text-amber-600 shrink-0">
                  <Layers className="h-3.5 w-3.5" />
                </span>
                <span>
                  <strong className="text-slate-800 block">3. Batched ViT + CLIP Inference</strong>
                  Classifies every object simultaneously and persists an audit trail to History.
                </span>
              </li>
            </ul>

            <div className="pt-3 border-t border-slate-100 space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Supported Materials
              </span>
              <div className="flex flex-wrap gap-1.5">
                {V2_WASTE_CLASSES.map((cls) => (
                  <CategoryBadge key={cls} category={cls} size="sm" />
                ))}
              </div>
            </div>
          </aside>
        </div>
      )}

      {error && (
        <StatusBanner variant="error" title="Detection Failed">
          {error}
        </StatusBanner>
      )}

      {result && (
        <div className="space-y-6">
          <RecyclingSummaryCard summary={result.recycling_summary} />

          <AnnotatedImageViewer
            imageBase64={result.clean_image_base64 || result.annotated_image_base64}
            imageWidth={result.image_width}
            imageHeight={result.image_height}
            objects={result.objects}
            selectedObjectId={selectedObjectId}
            onSelectObject={(id) => setSelectedObjectId(id)}
            materialFilter={materialFilter}
            onFilterChange={(f) => setMaterialFilter(f)}
          />

          {selectedObject && (
            <ObjectDetailPanel
              object={selectedObject}
              scanId={result.scan_id ?? undefined}
              onClose={() => setSelectedObjectId(null)}
            />
          )}

          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-base font-heading font-extrabold text-slate-900">
                Itemized Detections ({displayedObjects.length}
                {materialFilter !== 'all' ? ` of ${result.objects.length}` : ''})
              </h3>
              <span className="text-xs text-slate-400">
                Click any bounding box or card to inspect ViT, CLIP, resin &amp; XAI attribution
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {displayedObjects.map((obj) => (
                <ObjectCard
                  key={obj.object_id}
                  object={obj}
                  isSelected={selectedObjectId === obj.object_id}
                  onSelect={() => setSelectedObjectId(obj.object_id)}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
