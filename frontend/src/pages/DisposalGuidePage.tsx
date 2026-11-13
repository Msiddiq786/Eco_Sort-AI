import { Compass, Sparkles } from 'lucide-react'
import PreparationSteps from '@/components/guide/PreparationSteps'
import MaterialSearch from '@/components/guide/MaterialSearch'
import HazardousWarning from '@/components/guide/HazardousWarning'

export default function DisposalGuidePage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
              <Sparkles className="h-3 w-3" />
              7 Waste Categories Supported
            </span>
          </div>
          <h1 className="text-2xl font-heading font-extrabold tracking-tight text-slate-900 flex items-center gap-2.5">
            <Compass className="h-6 w-6 text-emerald-600" />
            Smart Recycling Guide
          </h1>
          <p className="text-slate-600 text-sm mt-1 max-w-2xl leading-relaxed">
            Searchable materials catalog, localized preparation checklists, contamination hazards, and proper disposal protocols for residential and commercial waste.
          </p>
        </div>
      </div>

      {/* 4 Golden Steps Protocol */}
      <PreparationSteps />

      {/* Hazardous Materials Safety Advisory */}
      <HazardousWarning />

      {/* Searchable Material Catalog */}
      <MaterialSearch />
    </div>
  )
}
