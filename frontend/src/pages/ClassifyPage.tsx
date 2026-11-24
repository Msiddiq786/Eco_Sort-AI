import { useState } from 'react'
import { ScanLine, BrainCircuit } from 'lucide-react'
import DropZone from '@/components/classify/DropZone'
import ResultCard from '@/components/classify/ResultCard'
import { XAIExplanationPanel } from '@/components/xai/XAIExplanationPanel'
import { predictImage, explainImage } from '@/services/api'
import type { PredictResponse, XAIExplanation } from '@/types'

export default function ClassifyPage() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [xai, setXai] = useState<XAIExplanation | null>(null)
  const [xaiLoading, setXaiLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastFile, setLastFile] = useState<File | null>(null)

  const handleFile = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setXai(null)
    setLastFile(file)
    const objUrl = URL.createObjectURL(file)
    setPreviewUrl(objUrl)
    try {
      const res = await predictImage(file)
      setResult(res)
      // If XAI was embedded in the response, use it
      if (res.xai_explanation) {
        setXai(res.xai_explanation)
      }
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'An unexpected error occurred.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!lastFile) return
    setXaiLoading(true)
    try {
      const explanation = await explainImage(lastFile, true)
      setXai(explanation)
    } catch (err) {
      console.warn('XAI explanation failed:', err)
    } finally {
      setXaiLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <ScanLine className="h-6 w-6 text-primary" />
          Classify Waste
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Upload an image to classify it using the ViT + CLIP ensemble.
        </p>
      </div>

      <DropZone onFile={handleFile} loading={loading} />

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive whitespace-pre-wrap">
          {error}
        </div>
      )}

      {result && <ResultCard result={result} imageUrl={previewUrl} />}

      {/* XAI Section */}
      {result && !xai && (
        <div className="flex justify-center">
          <button
            onClick={handleExplain}
            disabled={xaiLoading}
            className="flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-5 py-2.5 text-sm font-medium text-zinc-700 shadow-sm hover:bg-zinc-50 disabled:opacity-50 transition-colors"
          >
            <BrainCircuit className="h-4 w-4 text-indigo-500" />
            {xaiLoading ? 'Generating explanation…' : '🧠 Explain This Prediction (XAI)'}
          </button>
        </div>
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
