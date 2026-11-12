import { useState } from 'react'
import { ScanLine } from 'lucide-react'
import DropZone from '@/components/classify/DropZone'
import ResultCard from '@/components/classify/ResultCard'
import { predictImage } from '@/services/api'
import type { PredictResponse } from '@/types'

export default function ClassifyPage() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFile = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    const objUrl = URL.createObjectURL(file)
    setPreviewUrl(objUrl)
    try {
      const res = await predictImage(file)
      setResult(res)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'An unexpected error occurred.'
      setError(msg)
    } finally {
      setLoading(false)
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
    </div>
  )
}
