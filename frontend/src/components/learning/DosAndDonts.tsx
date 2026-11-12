import { CheckCircle2, XCircle } from 'lucide-react'

export const DOS_LIST = [
  {
    title: 'Rinse food containers clean',
    detail: 'Give bottles, jars, and cans a quick rinse to eliminate grease, sauces, and sugar residue.',
  },
  {
    title: 'Keep recyclables loose in the bin',
    detail: 'Never tie recyclables inside plastic trash bags — optical sorters reject bagged recyclables.',
  },
  {
    title: 'Flatten corrugated cardboard boxes',
    detail: 'Collapse boxes flat to conserve bin space and prevent conveyor jams at sorting facilities.',
  },
  {
    title: 'Leave caps attached to rigid bottles',
    detail: 'Modern recycling facilities prefer plastic bottle caps tightly screwed onto empty crushed bottles.',
  },
  {
    title: 'Verify local municipality rules',
    detail: 'When in doubt, check your city council or waste contractor guidelines for accepted plastic resin codes.',
  },
]

export const DONTS_LIST = [
  {
    title: "Don't 'Wishcycle'",
    detail: 'Hoping an item is recyclable does not make it so. Contaminated batches end up in landfills.',
  },
  {
    title: 'Never recycle greasy pizza boxes',
    detail: 'Grease and oil soak into cardboard fibers and cannot be separated during water-based pulping.',
  },
  {
    title: 'Keep flexible plastic bags out of curbside bins',
    detail: 'Grocery shopping bags, bubble wrap, and cling film wrap around spinning gears and jam plant machinery.',
  },
  {
    title: 'Never mix drinking glassware or ceramics with bottles',
    detail: 'Mugs, Pyrex cookware, and mirrors have higher melting points and cause structural flaws in new glass.',
  },
  {
    title: 'Keep hazardous materials and batteries out',
    detail: 'Lithium-ion batteries cause severe fires when crushed inside waste collection trucks.',
  },
]

export default function DosAndDonts() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* DO's */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-emerald-600 text-white">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-heading font-bold text-lg text-emerald-950">Recycling Do's</h3>
            <p className="text-xs text-emerald-800">Essential best practices for pure recycling streams</p>
          </div>
        </div>

        <ul className="space-y-3.5">
          {DOS_LIST.map((item, idx) => (
            <li key={idx} className="flex items-start gap-3 text-sm">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-200/80 text-[11px] font-bold text-emerald-900 mt-0.5">
                ✓
              </span>
              <div>
                <p className="font-semibold text-emerald-950 text-xs sm:text-sm">{item.title}</p>
                <p className="text-xs text-emerald-800/90 leading-relaxed mt-0.5">{item.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>

      {/* DONT's */}
      <div className="rounded-xl border border-rose-200 bg-rose-50/40 p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-rose-600 text-white">
            <XCircle className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-heading font-bold text-lg text-rose-950">Recycling Don'ts</h3>
            <p className="text-xs text-rose-800">High-risk actions that contaminate and spoil recycling loads</p>
          </div>
        </div>

        <ul className="space-y-3.5">
          {DONTS_LIST.map((item, idx) => (
            <li key={idx} className="flex items-start gap-3 text-sm">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-rose-200/80 text-[11px] font-bold text-rose-900 mt-0.5">
                ✗
              </span>
              <div>
                <p className="font-semibold text-rose-950 text-xs sm:text-sm">{item.title}</p>
                <p className="text-xs text-rose-800/90 leading-relaxed mt-0.5">{item.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
