import { useState } from 'react'
import type { WasteCategory } from '@/types'
import { logConfirmedActivity } from '@/services/ecoStorage'
import { PlusCircle, X, ShieldCheck, Sparkles } from 'lucide-react'

interface LogActivityModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

const PRESET_ACTIONS = [
  {
    title: 'Rinsed & Deposited Rigid Plastic Containers',
    category: 'plastic' as WasteCategory,
    points: 20,
    note: 'Cleaned soda/water bottle and screwed cap tightly',
  },
  {
    title: 'Broke Down & Flattened Corrugated Cardboard',
    category: 'cardboard' as WasteCategory,
    points: 25,
    note: 'Removed plastic tape and flattened box for recycling',
  },
  {
    title: 'Cleaned Aluminum / Tin Food Can',
    category: 'metal' as WasteCategory,
    points: 20,
    note: 'Washed out food residue to avoid pest contamination',
  },
  {
    title: 'Sorted & Bundled Clean Paper / Newspaper',
    category: 'paper' as WasteCategory,
    points: 15,
    note: 'Kept documents dry without staples or binder clips',
  },
  {
    title: 'Safely Deposited Glass Bottles at Recycling Hub',
    category: 'glass' as WasteCategory,
    points: 30,
    note: 'Intact glass containers dropped in bottle bank',
  },
  {
    title: 'Composted Organic Vegetable & Fruit Peels',
    category: 'trash' as WasteCategory,
    points: 25,
    note: 'Diverted organic food waste to home/municipal compost',
  },
]

export default function LogActivityModal({ isOpen, onClose, onSuccess }: LogActivityModalProps) {
  const [selectedPreset, setSelectedPreset] = useState<number | null>(0)
  const customTitle = ''
  const [category, setCategory] = useState<WasteCategory>('plastic')
  const [confirmed, setConfirmed] = useState(false)
  const [points, setPoints] = useState(20)

  if (!isOpen) return null

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!confirmed) return

    const title = selectedPreset !== null ? PRESET_ACTIONS[selectedPreset].title : customTitle || 'Confirmed Waste Segregation'
    const finalCat = selectedPreset !== null ? PRESET_ACTIONS[selectedPreset].category : category
    const finalPoints = selectedPreset !== null ? PRESET_ACTIONS[selectedPreset].points : points

    logConfirmedActivity({
      title,
      category: finalCat,
      points: finalPoints,
      description: 'Confirmed user recycling activity logged into EcoSort AI.',
    })

    onSuccess()
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl border border-slate-100 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-100 text-emerald-800">
              <PlusCircle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-base text-slate-900">
                Log Confirmed Recycling Activity
              </h3>
              <p className="text-[11px] text-muted-foreground">
                Earn verified Eco Points for real-world waste segregation actions
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
          {/* Presets */}
          <div>
            <label className="block font-semibold text-slate-700 mb-2">
              Select Recycling Action:
            </label>
            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {PRESET_ACTIONS.map((preset, idx) => (
                <button
                  type="button"
                  key={idx}
                  onClick={() => {
                    setSelectedPreset(idx)
                    setCategory(preset.category)
                    setPoints(preset.points)
                  }}
                  className={`w-full text-left p-3 rounded-lg border text-xs transition-all flex items-center justify-between ${
                    selectedPreset === idx
                      ? 'border-emerald-500 bg-emerald-50/70 text-emerald-950 font-medium'
                      : 'border-slate-200 hover:border-slate-300 bg-slate-50/50 text-slate-800'
                  }`}
                >
                  <div>
                    <p className="font-semibold">{preset.title}</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">{preset.note}</p>
                  </div>
                  <span className="shrink-0 px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold text-[10px]">
                    +{preset.points} pts
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* User Confirmation Checkbox */}
          <div className="rounded-xl bg-slate-50 border border-slate-200 p-3.5 space-y-2">
            <div className="flex items-start gap-2.5">
              <input
                type="checkbox"
                id="confirmRecycle"
                checked={confirmed}
                onChange={e => setConfirmed(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
              />
              <label htmlFor="confirmRecycle" className="text-xs text-slate-700 leading-snug cursor-pointer">
                <strong>I confirm that I have physically performed this recycling activity</strong> and followed clean segregation guidelines (rinsed, flattened, or deposited into the proper bin).
              </label>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-700 pt-1">
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>EcoSort AI rewards verified human stewardship, not unconfirmed automated scans.</span>
            </div>
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
              disabled={!confirmed}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow-sm flex items-center gap-1.5 transition-colors"
            >
              <Sparkles className="h-4 w-4" /> Confirm & Claim Points
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
