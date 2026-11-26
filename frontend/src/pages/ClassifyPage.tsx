import { useState } from 'react'
import {
  ScanLine,
  BrainCircuit,
  Sparkles,
  RotateCcw,
  Cpu,
  Layers,
  Recycle,
  FlaskConical,
} from 'lucide-react'
import DropZone from '@/components/classify/DropZone'
import ResultCard from '@/components/classify/ResultCard'
import { XAIExplanationPanel } from '@/components/xai/XAIExplanationPanel'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
import CategoryBadge from '@/components/ui/CategoryBadge'
import { predictImage, explainImage } from '@/services/api'
import { V2_WASTE_CLASSES, type PredictResponse, type XAIExplanation } from '@/types'

export default function ClassifyPage() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [xai, setXai] = useState<XAIExplanation | null>(null)
  const [xaiLoading, setXaiLoading] = useState(false)
  const [xaiError, setXaiError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastFile, setLastFile] = useState<File | null>(null)

  const handleFile = async (file: File) => {
    setLoading(true)
    setError(null)
    setXaiError(null)
    setResult(null)
    setXai(null)
    setLastFile(file)
    const objUrl = URL.createObjectURL(file)
    setPreviewUrl(objUrl)
    try {
      const res = await predictImage(file)
      setResult(res)
      if (res.xai_explanation) {
        setXai(res.xai_explanation)
      }
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'An unexpected error occurred while classifying the image.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!lastFile) return
    setXaiLoading(true)
    setXaiError(null)
    try {
      const explanation = await explainImage(lastFile, true)
      setXai(explanation)
    } catch (err) {
      console.warn('XAI explanation failed:', err)
      setXaiError('Could not generate visual XAI explanation for this item.')
    } finally {
      setXaiLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setXai(null)
    setError(null)
    setXaiError(null)
    setLastFile(null)
    setPreviewUrl(null)
  }

  return (
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="Dual-Model Ensemble"
        eyebrowIcon={Sparkles}
        badge="7-Class Taxonomy"
        icon={ScanLine}
        title="Classify Waste"
        subtitle="Upload or capture a waste item image to classify it using fine-tuned ViT-Base + zero-shot CLIP V2 embeddings, resin code screening, and Smart Bin guidance."
        actions={
          result ? (
            <button
              type="button"
              onClick={handleReset}
              className="btn-secondary text-xs px-3.5 py-2"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Classify Another</span>
            </button>
          ) : null
        }
      />

      {!result ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-8">
            <DropZone onFile={handleFile} loading={loading} />
          </div>

          <aside className="lg:col-span-4 space-y-4">
            <div className="card-eco p-5 space-y-4">
              <h2 className="font-heading font-bold text-sm text-slate-900">
                Classification Pipeline
              </h2>
              <ul className="space-y-3 text-xs text-slate-600">
                <li className="flex items-start gap-2.5">
                  <span className="p-1.5 rounded-lg bg-emerald-50 text-ecosort-emerald shrink-0">
                    <Cpu className="h-3.5 w-3.5" />
                  </span>
                  <span>
                    <strong className="text-slate-800 block">ViT-Base Patch16</strong>
                    Fine-tuned transformer captures fine texture & material cues.
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="p-1.5 rounded-lg bg-sky-50 text-sky-600 shrink-0">
                    <Layers className="h-3.5 w-3.5" />
                  </span>
                  <span>
                    <strong className="text-slate-800 block">CLIP Foundation V2</strong>
                    Semantic class centroids add zero-shot organic & material coverage.
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600 shrink-0">
                    <FlaskConical className="h-3.5 w-3.5" />
                  </span>
                  <span>
                    <strong className="text-slate-800 block">Resin &amp; Hazard Screen</strong>
                    Identifies ASTM #1–#7 polymers, batteries, and pressurized aerosols.
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="p-1.5 rounded-lg bg-lime-50 text-lime-700 shrink-0">
                    <Recycle className="h-3.5 w-3.5" />
                  </span>
                  <span>
                    <strong className="text-slate-800 block">SWM Preparation Rules</strong>
                    Outputs bin stream, rinsing steps, and contamination warnings.
                  </span>
                </li>
              </ul>

              <div className="pt-3 border-t border-slate-100 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Supported Classes
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {V2_WASTE_CLASSES.map((cls) => (
                    <CategoryBadge key={cls} category={cls} size="sm" />
                  ))}
                </div>
              </div>
            </div>
          </aside>
        </div>
      ) : (
        <DropZone onFile={handleFile} loading={loading} />
      )}

      {error && (
        <StatusBanner variant="error" title="Classification Error">
          <span className="whitespace-pre-wrap">{error}</span>
        </StatusBanner>
      )}

      {result && <ResultCard result={result} imageUrl={previewUrl} />}

      {result && !xai && (
        <div className="flex justify-center pt-1">
          <button
            type="button"
            onClick={handleExplain}
            disabled={xaiLoading}
            className="btn-secondary"
          >
            <BrainCircuit className="h-4 w-4 text-ecosort-emerald" />
            <span>
              {xaiLoading ? 'Generating explanation…' : 'Explain This Prediction (XAI)'}
            </span>
          </button>
        </div>
      )}

      {xaiError && (
        <StatusBanner variant="warning" title="Explanation Unavailable">
          {xaiError}
        </StatusBanner>
      )}

      {xai && (
        <XAIExplanationPanel
          explanation={xai}
          originalImageSrc={previewUrl ?? undefined}
          confidenceStatus={result?.confidence_status ?? 'normal'}
        />
      )}
    </div>
  )
}
