import { Compass } from 'lucide-react'
import PreparationSteps from '@/components/guide/PreparationSteps'
import MaterialSearch from '@/components/guide/MaterialSearch'
import HazardousWarning from '@/components/guide/HazardousWarning'

export default function DisposalGuidePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Compass className="h-6 w-6 text-primary" />
          Smart Waste Disposal Guide
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Complete recycling protocol, step-by-step preparation tips, and hazardous material safety guidelines.
        </p>
      </div>

      {/* 4 Golden Steps */}
      <PreparationSteps />

      {/* Hazardous Materials Callout */}
      <HazardousWarning />

      {/* Searchable Material Catalog */}
      <MaterialSearch />
    </div>
  )
}
