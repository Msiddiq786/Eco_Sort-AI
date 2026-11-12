import { Droplets, Wind, Scissors, Unlock } from 'lucide-react'

export const PREPARATION_STEPS = [
  {
    step: 1,
    title: 'Empty & Scrape',
    description: 'Ensure food containers, jars, and bottles are completely empty. Scrape off remaining pastes or peanut butter.',
    icon: Droplets,
  },
  {
    step: 2,
    title: 'Quick Rinse & Dry',
    description: 'Swirl with a small splash of leftover dishwater. Items do not need to be dishwasher-sterile, but must be free of puddles.',
    icon: Wind,
  },
  {
    step: 3,
    title: 'Disassemble & Flatten',
    description: 'Remove plastic films, detach metal lids from glass jars, and flatten all cardboard boxes to maximize bin capacity.',
    icon: Scissors,
  },
  {
    step: 4,
    title: 'Keep Loose in Bin',
    description: 'Never bundle or tie recyclables inside plastic garbage bags. Automated optical sorting machines cannot open bags.',
    icon: Unlock,
  },
]

export default function PreparationSteps() {
  return (
    <div className="rounded-xl border border-emerald-200 bg-white p-6 shadow-sm space-y-4">
      <div>
        <h3 className="font-heading font-bold text-lg text-emerald-950">
          The 4 Golden Preparation Steps
        </h3>
        <p className="text-xs text-muted-foreground">
          Follow these simple steps before placing any recyclable into collection bins
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {PREPARATION_STEPS.map(s => {
          const Icon = s.icon
          return (
            <div
              key={s.step}
              className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-100 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2 rounded-lg bg-emerald-600 text-white">
                    <Icon className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-200/80 text-emerald-900">
                    Step {s.step}
                  </span>
                </div>
                <h4 className="font-heading font-bold text-sm text-emerald-950 mb-1">
                  {s.title}
                </h4>
                <p className="text-xs text-emerald-900/80 leading-relaxed">
                  {s.description}
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
