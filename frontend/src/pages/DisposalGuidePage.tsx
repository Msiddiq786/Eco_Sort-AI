import { Compass, Sparkles } from 'lucide-react'
import PageHeader from '@/components/ui/PageHeader'
import PreparationSteps from '@/components/guide/PreparationSteps'
import MaterialSearch from '@/components/guide/MaterialSearch'
import HazardousWarning from '@/components/guide/HazardousWarning'

export default function DisposalGuidePage() {
  return (
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="7 Waste Categories Supported"
        eyebrowIcon={Sparkles}
        badge="SWM Protocol"
        icon={Compass}
        title="Smart Recycling Guide"
        subtitle="Searchable materials catalog, localized preparation checklists, contamination hazards, and proper disposal protocols for household and commercial waste."
      />

      <PreparationSteps />

      <HazardousWarning />

      <MaterialSearch />
    </div>
  )
}
