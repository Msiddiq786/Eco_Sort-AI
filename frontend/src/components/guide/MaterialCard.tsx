import React from 'react'
import {
  ArrowRight,
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
} from 'lucide-react'
import type { RecyclingGuideMaterial } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'

interface MaterialCardProps {
  material: RecyclingGuideMaterial
  onSelect: (material: RecyclingGuideMaterial) => void
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

export default function MaterialCard({ material, onSelect }: MaterialCardProps) {
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
      onClick={() => onSelect(material)}
      className="group relative rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm hover:shadow-md hover:border-emerald-300 transition-all duration-200 flex flex-col justify-between cursor-pointer"
    >
      <div>
        {/* Header row: Icon, Category Pill, Status Badge */}
        <div className="flex items-start justify-between gap-3 mb-3.5">
          <div className="flex items-center gap-3">
            <div
              className="flex h-11 w-11 items-center justify-center rounded-xl text-white shadow-sm shrink-0 group-hover:scale-105 transition-transform"
              style={{ background: categoryColor }}
            >
              <IconComponent className="h-5 w-5" />
            </div>
            <div>
              <span
                className="text-[10px] font-bold px-2 py-0.5 rounded-full text-white uppercase tracking-wider inline-block mb-1"
                style={{ background: categoryColor }}
              >
                {capitalize(material.category)}
              </span>
              <h3 className="font-heading font-bold text-base text-slate-900 leading-snug group-hover:text-emerald-800 transition-colors">
                {material.name}
              </h3>
            </div>
          </div>
        </div>

        {/* Recycling Status Badge */}
        <div className="mb-3">
          <span
            className={`inline-block text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${badgeClass}`}
          >
            {material.recyclingPossibility}
          </span>
        </div>

        {/* Brief description */}
        <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed mb-3">
          {material.description}
        </p>

        {/* Examples preview */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {material.commonExamples.slice(0, 3).map((ex, idx) => (
            <span
              key={idx}
              className="text-[10px] font-medium bg-slate-100 text-slate-700 px-2 py-0.5 rounded-md border border-slate-200/60"
            >
              {ex}
            </span>
          ))}
          {material.commonExamples.length > 3 && (
            <span className="text-[10px] font-semibold text-slate-400 self-center">
              +{material.commonExamples.length - 3} more
            </span>
          )}
        </div>
      </div>

      {/* Card Footer: Bin and CTA */}
      <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 min-w-0 pr-2">
          <span
            className="h-2.5 w-2.5 rounded-full shrink-0"
            style={{ background: material.disposalMethod.binColor }}
          />
          <span className="text-[11px] font-medium text-slate-600 truncate">
            {material.disposalMethod.binName}
          </span>
        </div>
        <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-700 group-hover:translate-x-0.5 transition-transform shrink-0">
          View Guide <ArrowRight className="h-3 w-3" />
        </span>
      </div>
    </div>
  )
}
