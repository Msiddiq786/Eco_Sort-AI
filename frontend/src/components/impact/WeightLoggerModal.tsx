import { useState } from 'react'
import type { WasteCategory } from '@/types'
import { WASTE_CLASSES, capitalize } from '@/lib'
import { addWeightLog } from '@/services/ecoStorage'
import { X, Scale, Sparkles } from 'lucide-react'

interface WeightLoggerModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export default function WeightLoggerModal({ isOpen, onClose, onSuccess }: WeightLoggerModalProps) {
  const [category, setCategory] = useState<WasteCategory>('plastic')
  const [weightValue, setWeightValue] = useState<string>('250')
  const [unit, setUnit] = useState<'g' | 'kg'>('g')
  const [note, setNote] = useState<string>('')

  if (!isOpen) return null

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const num = parseFloat(weightValue)
    if (isNaN(num) || num <= 0) return

    const weightGrams = unit === 'kg' ? Math.round(num * 1000) : Math.round(num)
    addWeightLog({
      category,
      weightGrams,
      note: note.trim() || undefined,
    })

    onSuccess()
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-md rounded-2xl bg-white p-6 shadow-xl border border-slate-100">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-emerald-100 text-emerald-800">
              <Scale className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-base text-slate-900">
                Log Diverted Waste Weight
              </h3>
              <p className="text-[11px] text-muted-foreground">
                Track exact quantities and calculate authentic environmental offsets
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4 text-xs sm:text-sm">
          {/* Category */}
          <div>
            <label className="block font-semibold text-slate-700 mb-1">
              Waste Material Category
            </label>
            <select
              value={category}
              onChange={e => setCategory(e.target.value as WasteCategory)}
              className="w-full rounded-lg border border-input bg-background p-2.5 text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {WASTE_CLASSES.map(c => (
                <option key={c} value={c}>
                  {capitalize(c)}
                </option>
              ))}
            </select>
          </div>

          {/* Weight & Unit */}
          <div>
            <label className="block font-semibold text-slate-700 mb-1">
              Quantity / Weight
            </label>
            <div className="flex gap-2">
              <input
                type="number"
                min="1"
                step="any"
                required
                value={weightValue}
                onChange={e => setWeightValue(e.target.value)}
                className="flex-1 rounded-lg border border-input bg-background p-2.5 text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="e.g. 500"
              />
              <select
                value={unit}
                onChange={e => setUnit(e.target.value as 'g' | 'kg')}
                className="rounded-lg border border-input bg-background px-3 py-2 text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary font-medium"
              >
                <option value="g">Grams (g)</option>
                <option value="kg">Kilograms (kg)</option>
              </select>
            </div>
          </div>

          {/* Note */}
          <div>
            <label className="block font-semibold text-slate-700 mb-1">
              Notes / Description (Optional)
            </label>
            <input
              type="text"
              value={note}
              onChange={e => setNote(e.target.value)}
              className="w-full rounded-lg border border-input bg-background p-2.5 text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary"
              placeholder="e.g. 4 flattened delivery cardboard boxes"
            />
          </div>

          <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-sm flex items-center gap-1.5 transition-colors"
            >
              <Sparkles className="h-4 w-4" /> Save & Calculate Impact
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
