import React from 'react'
import {
  X,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Leaf,
  Info,
  Package,
  Box,
  Wine,
  Hammer,
  Shield,
  FileText,
  BookOpen,
  Receipt,
  Milk,
  Droplets,
  ShoppingBag,
  Apple,
  Trash2,
  BatteryCharging,
  Pizza,
} from 'lucide-react'
import type { RecyclingGuideMaterial } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'

interface MaterialDetailModalProps {
  material: RecyclingGuideMaterial | null
  onClose: () => void
}

const ICON_MAP: Record<string, React.ElementType> = {
  Package,
  Box,
  Wine,
  Hammer,
  Shield,
  FileText,
  BookOpen,
  Receipt,
  Milk,
  Droplets,
  ShoppingBag,
  Apple,
  Leaf,
  Trash2,
  BatteryCharging,
  Pizza,
  AlertTriangle,
}

export default function MaterialDetailModal({
  material,
  onClose,
}: MaterialDetailModalProps) {
  if (!material) return null

  const IconComponent =
    (material.iconName && ICON_MAP[material.iconName]) || Package
  const categoryColor =
    CATEGORY_COLORS[material.category] ||
    (material.category === 'hazardous' ? '#EF4444' : '#6B7280')

  const badgeColorMap: Record<string, string> = {
    'Widely Recyclable': 'bg-emerald-100 text-emerald-800 border-emerald-300',
    'Conditionally Recyclable': 'bg-amber-100 text-amber-800 border-amber-300',
    'Compostable': 'bg-lime-100 text-lime-800 border-lime-300',
    'Special Disposal Required': 'bg-sky-100 text-sky-800 border-sky-300',
    'Residual Waste (Non-Recyclable)': 'bg-slate-100 text-slate-800 border-slate-300',
  }

  const badgeClass =
    badgeColorMap[material.recyclingPossibility] ||
    'bg-slate-100 text-slate-800 border-slate-300'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl rounded-2xl bg-white shadow-2xl border border-slate-100 max-h-[92vh] flex flex-col overflow-hidden my-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-slate-100 bg-slate-50/50">
          <div className="flex items-start gap-4">
            <div
              className="flex h-12 w-12 items-center justify-center rounded-2xl text-white shadow-md shrink-0"
              style={{ background: categoryColor }}
            >
              <IconComponent className="h-6 w-6" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded-full text-white uppercase tracking-wider"
                  style={{ background: categoryColor }}
                >
                  {capitalize(material.category)}
                </span>
                <span
                  className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${badgeClass}`}
                >
                  {material.recyclingPossibility}
                </span>
              </div>
              <h2 className="text-xl font-heading font-extrabold text-slate-900">
                {material.name}
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-xl p-2 text-slate-400 hover:bg-slate-200/60 hover:text-slate-700 transition"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="overflow-y-auto p-6 space-y-6 text-sm">
          {/* Description */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1.5">
              Material Description & Lifecycle
            </h3>
            <p className="text-slate-700 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200/70">
              {material.description}
            </p>
          </div>

          {/* Common Examples */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
              Common Examples
            </h3>
            <div className="flex flex-wrap gap-2">
              {material.commonExamples.map((ex, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center px-3 py-1 rounded-lg text-xs font-medium bg-slate-100 text-slate-800 border border-slate-200"
                >
                  {ex}
                </span>
              ))}
            </div>
          </div>

          {/* Recommended Disposal Method & Bin */}
          <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-sm space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Recommended Disposal Method
            </h3>
            <div className="flex items-center gap-2.5">
              <span
                className="h-3.5 w-3.5 rounded-full shrink-0 shadow-sm"
                style={{ background: material.disposalMethod.binColor }}
              />
              <span className="font-heading font-bold text-slate-900 text-sm">
                {material.disposalMethod.binName}
              </span>
            </div>
            <p className="text-xs text-slate-600 leading-normal pl-6">
              {material.disposalMethod.instructions}
            </p>
          </div>

          {/* Preparation Instructions Checklist */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
              Step-by-Step Preparation Protocol
            </h3>
            <ul className="space-y-2">
              {material.preparationInstructions.map((step, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2.5 p-2.5 rounded-lg bg-emerald-50/50 border border-emerald-100 text-slate-800 text-xs"
                >
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>{step}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Avoid Mixing / Contamination Hazards */}
          {material.avoidMixing && material.avoidMixing.length > 0 && (
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-rose-700 flex items-center gap-1.5 mb-2">
                <AlertTriangle className="h-4 w-4 text-rose-600" />
                Items to Avoid Mixing (Contamination Hazards)
              </h3>
              <ul className="space-y-2">
                {material.avoidMixing.map((item, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2.5 p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-950 text-xs"
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-rose-500 shrink-0 mt-1.5" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Environmental Benefits */}
          {material.environmentalBenefits && (
            <div className="p-4 rounded-xl bg-lime-50/60 border border-lime-200 text-xs text-slate-800 space-y-1">
              <div className="flex items-center gap-1.5 text-lime-900 font-bold">
                <Leaf className="h-4 w-4 text-lime-700" />
                <span>Verified Environmental Impact</span>
              </div>
              <p className="leading-relaxed text-slate-700">
                {material.environmentalBenefits}
              </p>
            </div>
          )}

          {/* Safety Warnings */}
          {material.safetyWarnings && (
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-950 space-y-1">
              <div className="flex items-center gap-1.5 text-amber-900 font-bold">
                <Flame className="h-4 w-4 text-amber-700" />
                <span>Safety Advisory</span>
              </div>
              <p className="leading-relaxed text-amber-900 font-medium">
                {material.safetyWarnings}
              </p>
            </div>
          )}

          {/* Municipal Variance Disclaimer */}
          <div className="flex items-start gap-2 text-[11px] text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200">
            <Info className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
            <span>
              <strong>Note:</strong> Waste sorting and collection protocols vary by local
              municipality, waste-to-energy capabilities, and Materials Recovery
              Facility (MRF) equipment. Always verify with your local sanitation guidelines.
            </span>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-slate-800 text-white text-xs font-semibold hover:bg-slate-900 transition shadow-sm"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  )
}
