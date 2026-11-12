import { useState } from 'react'
import { Crosshair, RefreshCw, AlertCircle } from 'lucide-react'
import DropZone from '@/components/classify/DropZone'
import AnnotatedImageViewer from '@/components/detect/AnnotatedImageViewer'
import ObjectCard from '@/components/detect/ObjectCard'
import ObjectDetailPanel from '@/components/detect/ObjectDetailPanel'
import RecyclingSummaryCard from '@/components/detect/RecyclingSummaryCard'
import { detectObjects } from '@/services/api'
import type { DetectResponse } from '@/types'

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
      if (res.objects.length > 0) {
        setSelectedObjectId(res.objects[0].object_id)
      }
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

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-heading font-extrabold flex items-center gap-2.5 text-slate-900">
              <Crosshair className="h-6 w-6 text-emerald-600" />
              Multi-Object Waste Detection
            </h1>
            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-xs font-bold uppercase tracking-wider">
              Beta
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">
            Detect, localize and segregate multiple waste items in a single scene with per-object AI classification.
          </p>
        </div>

        {result && (
          <button
            type="button"
            onClick={() => {
              setResult(null)
              setSelectedObjectId(null)
            }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-700 shadow-xs transition-colors self-start sm:self-auto"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Upload Another Scene
          </button>
        )}
      </div>

      {/* Upload Zone (when no result) */}
      {!result && (
        <div className="space-y-4">
          <DropZone onFile={handleFile} loading={loading} />
          <div className="rounded-xl border border-slate-200/80 bg-slate-50/60 p-4 text-xs text-slate-500 leading-relaxed">
            <strong className="text-slate-700">Supported Multi-Object Scenes:</strong> Upload photos with multiple items (e.g. bottles, cans, paper, boxes, vegetable scraps). The system automatically isolates individual bounding boxes, classifies each crop using the ViT + CLIP ensemble, and outputs itemized recycling rules.
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 flex items-start gap-3 text-rose-800 text-sm">
          <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold">Detection Failed</h4>
            <p className="text-xs text-rose-700 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div className="space-y-6">
          {/* Recycling Summary */}
          <RecyclingSummaryCard summary={result.recycling_summary} />

          {/* Interactive Bounding Box Viewer */}
          <AnnotatedImageViewer
            imageBase64={result.annotated_image_base64}
            imageWidth={result.image_width}
            imageHeight={result.image_height}
            objects={result.objects}
            selectedObjectId={selectedObjectId}
            onSelectObject={(id) => setSelectedObjectId(id)}
            materialFilter={materialFilter}
            onFilterChange={(f) => setMaterialFilter(f)}
          />

          {/* Selected Object Detail Panel */}
          {selectedObject && (
            <ObjectDetailPanel
              object={selectedObject}
              onClose={() => setSelectedObjectId(null)}
            />
          )}

          {/* Detected Objects Grid */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-heading font-extrabold text-slate-900">
                Itemized Detections ({result.objects.length})
              </h3>
              <span className="text-xs text-slate-400">Click any card to inspect individual AI predictions</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.objects.map((obj) => (
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
