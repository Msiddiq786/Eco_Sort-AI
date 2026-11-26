import { useEffect, useState } from 'react'
import { CheckCircle2, Flag, Loader2, Send } from 'lucide-react'
import { submitFeedback } from '@/services/api'
import { V2_WASTE_CLASSES, type FeedbackResponse } from '@/types'
import { capitalize } from '@/lib/utils'

interface FeedbackCorrectionProps {
  predictedCategory: string
  predictionId?: number
  detectionRecordId?: number
  objectId?: number
  source: 'single' | 'multi_object' | 'history'
  imageBase64?: string | null
  onSubmitted?: (response: FeedbackResponse) => void
}

export default function FeedbackCorrection({
  predictedCategory,
  predictionId,
  detectionRecordId,
  objectId,
  source,
  imageBase64,
  onSubmitted,
}: FeedbackCorrectionProps) {
  const normalizedPredicted = predictedCategory.toLowerCase()
  const [open, setOpen] = useState(false)
  const [corrected, setCorrected] = useState<string>(normalizedPredicted)
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [savedMsg, setSavedMsg] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    setOpen(false)
    setCorrected(predictedCategory.toLowerCase())
    setNotes('')
    setSavedMsg(null)
    setErrorMsg(null)
  }, [predictedCategory, predictionId, detectionRecordId, objectId])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setErrorMsg(null)
    try {
      const resp = await submitFeedback({
        predicted_category: normalizedPredicted,
        corrected_category: corrected,
        prediction_id: predictionId,
        detection_record_id: detectionRecordId,
        object_id: objectId,
        source,
        notes: notes.trim() || undefined,
        image_base64: imageBase64 ?? undefined,
      })
      setSavedMsg(resp.message)
      setOpen(false)
      onSubmitted?.(resp)
    } catch {
      setErrorMsg('Could not save feedback correction. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="rounded-xl border border-slate-200/90 bg-slate-50/70 p-3.5 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-slate-700 font-medium">
          <Flag className="h-3.5 w-3.5 text-amber-600 shrink-0" />
          <span>Spot a misclassification? Curate this sample for retraining.</span>
        </div>
        {!open && !savedMsg && (
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-900 hover:bg-amber-100 transition-colors"
          >
            Correct Label
          </button>
        )}
      </div>

      {savedMsg && (
        <div className="mt-2 flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-2.5 py-1.5 text-[11px] font-medium text-emerald-800">
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
          <span>{savedMsg}</span>
        </div>
      )}

      {open && (
        <form onSubmit={handleSubmit} className="mt-3 pt-3 border-t border-slate-200 space-y-2.5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <div>
              <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                Actual Material Class
              </label>
              <select
                aria-label="Actual Material Class"
                value={corrected}
                onChange={(e) => setCorrected(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              >
                {V2_WASTE_CLASSES.map((cls) => (
                  <option key={cls} value={cls}>
                    {capitalize(cls)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                Optional Note
              </label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g. Clear PET bottle, foil wrapper…"
                maxLength={250}
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>
          </div>

          {errorMsg && <p className="text-[11px] text-rose-600 font-medium">{errorMsg}</p>}

          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1 text-[11px] font-semibold text-white hover:bg-emerald-700 disabled:opacity-60"
            >
              {submitting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Send className="h-3.5 w-3.5" />
              )}
              Save Correction
            </button>
          </div>
        </form>
      )}
    </div>
  )
}
