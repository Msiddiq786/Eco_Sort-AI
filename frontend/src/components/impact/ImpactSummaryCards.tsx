import type { ImpactSummary } from '@/types'
import { Leaf, Droplets, Trees, Trash, Scale } from 'lucide-react'

interface ImpactSummaryCardsProps {
  summary: ImpactSummary
}

export default function ImpactSummaryCards({ summary }: ImpactSummaryCardsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {/* Total Weight */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Total Diverted</span>
          <Scale className="h-4 w-4 text-emerald-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-slate-800">
          {summary.totalWeightKg} <span className="text-sm font-normal text-slate-500">kg</span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Diverted from municipal landfills
        </p>
      </div>

      {/* CO2e Saved */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Est. GHG Avoided</span>
          <Leaf className="h-4 w-4 text-emerald-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-emerald-700">
          {summary.co2SavedKg} <span className="text-sm font-normal text-emerald-600">kg CO₂e</span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Equivalent to driving ~{(summary.co2SavedKg * 4.1).toFixed(0)} km less
        </p>
      </div>

      {/* Water Conserved */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Water Conserved</span>
          <Droplets className="h-4 w-4 text-blue-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-blue-700">
          {summary.waterSavedLiters} <span className="text-sm font-normal text-blue-600">L</span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Saved vs virgin raw material processing
        </p>
      </div>

      {/* Trees Preserved */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Trees Preserved</span>
          <Trees className="h-4 w-4 text-teal-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-teal-700">
          {summary.treesSaved} <span className="text-sm font-normal text-teal-600">trees</span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Via paper & cardboard recycling
        </p>
      </div>

      {/* Landfill Volume Saved */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="flex items-center justify-between text-muted-foreground mb-1 text-xs">
          <span>Landfill Volume Saved</span>
          <Trash className="h-4 w-4 text-amber-600" />
        </div>
        <div className="text-2xl font-heading font-bold text-amber-700">
          {summary.landfillDivertedM3} <span className="text-sm font-normal text-amber-600">m³</span>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Conserved compacted volume
        </p>
      </div>
    </div>
  )
}
