import { AlertOctagon, ArrowRight } from 'lucide-react'

export const COMMON_MISTAKES = [
  {
    mistake: 'Recycling Greasy Pizza Boxes',
    whyBad: 'Paper recycling uses water to slurry fibers. Oil does not mix with water, causing grease spots that weaken new cardboard batches.',
    correction: 'Tear off and recycle clean top lids. Compost or throw oily bottoms into residual trash.',
    icon: '🍕',
  },
  {
    mistake: 'Putting Plastic Bags in Curbside Bins',
    whyBad: 'Thin film plastics wrap around rotating sorter screens, halting plant operations for hours while workers manually cut them out.',
    correction: 'Return grocery shopping bags and film wrap to dedicated supermarket collection drop-off bins.',
    icon: '🛍️',
  },
  {
    mistake: 'Recycling Paper Coffee Cups',
    whyBad: 'Almost all take-away coffee cups are lined with a tightly bonded polyethylene plastic film to prevent leaks, making them non-repulpable.',
    correction: 'Dispose of coffee cups in residual trash or bring a reusable stainless steel thermal mug.',
    icon: '☕',
  },
  {
    mistake: 'Mixing Drinking Glasses & Pyrex with Bottles',
    whyBad: 'Drinkware, ceramics, mirrors, and oven glassware have higher melting points and contain chemicals that ruin container glass furnaces.',
    correction: 'Donate intact kitchenware to thrift stores; wrap broken ceramics and discard in trash.',
    icon: '🥛',
  },
  {
    mistake: 'Leaving Food Residue in Peanut Butter Jars',
    whyBad: 'Organic remnants rot in storage bales, emit foul odors, attract vermin, and can result in whole truckloads being sent to landfill.',
    correction: 'Use a spatula to scrape residue clean and give a quick warm water shake before binning.',
    icon: '🥜',
  },
  {
    mistake: 'Shredded Paper Thrown Loose in the Bin',
    whyBad: 'Loose paper strips fall through the screens at sorting facilities and end up contaminating glass or falling into landfill chutes.',
    correction: 'Place shredded paper in a paper bag before disposal, or use it for compost/mulch.',
    icon: '📄',
  },
]

export default function CommonMistakes() {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <AlertOctagon className="h-5 w-5 text-amber-600" />
        <div>
          <h3 className="font-heading font-bold text-lg text-slate-900">
            Top 6 "Wishcycling" Mistakes
          </h3>
          <p className="text-xs text-muted-foreground">
            Items people mistakenly throw in recycling bins that cause contamination
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {COMMON_MISTAKES.map((item, idx) => (
          <div
            key={idx}
            className="rounded-xl border border-amber-200/70 bg-white p-4 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{item.icon}</span>
                <h4 className="font-semibold text-slate-800 text-sm leading-snug">
                  {item.mistake}
                </h4>
              </div>

              <div className="rounded-lg bg-amber-50/70 p-2.5 text-xs text-amber-950 mb-3 border border-amber-100">
                <span className="font-semibold block text-[11px] text-amber-800 mb-0.5">
                  Why it causes trouble:
                </span>
                {item.whyBad}
              </div>
            </div>

            <div className="pt-2 border-t border-slate-100 flex items-start gap-1.5 text-xs text-emerald-800">
              <ArrowRight className="h-4 w-4 shrink-0 text-emerald-600 mt-0.5" />
              <span>
                <strong className="text-emerald-950">Better solution:</strong> {item.correction}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
