import { ShieldAlert, AlertTriangle, BatteryCharging, Flame, Biohazard, HelpCircle } from 'lucide-react'

export const HAZARDOUS_ITEMS = [
  {
    name: 'Lithium & Rechargeable Batteries',
    risk: 'Explosive thermal fire risk when crushed in trucks or conveyor rollers.',
    disposal: 'Drop off at hardware store or designated battery recycling kiosk.',
    icon: BatteryCharging,
  },
  {
    name: 'Propane & Butane Tanks',
    risk: 'Explosion hazard even when seemingly empty.',
    disposal: 'Return to retailer tank exchange or hazardous waste depot.',
    icon: Flame,
  },
  {
    name: 'Fluorescent Bulbs & Tubes',
    risk: 'Contain toxic mercury vapor that releases when broken.',
    disposal: 'Take to home improvement store or municipal hazardous waste center.',
    icon: AlertTriangle,
  },
  {
    name: 'Chemicals, Motor Oil & Paint',
    risk: 'Contaminates groundwater and ruins processing machinery.',
    disposal: 'Deliver sealed to your local Household Hazardous Waste (HHW) facility.',
    icon: Biohazard,
  },
]

export default function HazardousWarning() {
  return (
    <div className="space-y-4">
      {/* High Alert Banner */}
      <div className="rounded-xl border border-rose-300 bg-rose-50/60 p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-rose-600 text-white shrink-0">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <div>
            <h3 className="font-heading font-bold text-base text-rose-950">
              Hazardous Materials Safety Warning
            </h3>
            <p className="text-xs text-rose-900 mt-1 leading-relaxed">
              <strong>NEVER place hazardous materials in standard household recycling or trash bins.</strong> These materials endanger collection workers, can cause catastrophic fires, and leach toxic chemicals into municipal water tables.
            </p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {HAZARDOUS_ITEMS.map((item, idx) => {
            const Icon = item.icon
            return (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-white border border-rose-200 shadow-xs flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center gap-2 text-rose-700 font-semibold text-xs mb-1">
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{item.name}</span>
                  </div>
                  <p className="text-[11px] text-rose-950 leading-snug">
                    <strong className="text-rose-900">Danger:</strong> {item.risk}
                  </p>
                </div>
                <div className="mt-2.5 pt-2 border-t border-rose-100 text-[11px] text-emerald-800 font-medium">
                  <strong>Safe Path:</strong> {item.disposal}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Local Acceptance Caveat */}
      <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4 flex items-start gap-3 text-xs text-amber-950">
        <HelpCircle className="h-5 w-5 shrink-0 text-amber-700 mt-0.5" />
        <div>
          <h4 className="font-semibold text-amber-900 mb-0.5">
            Local Municipal Acceptance Caveats
          </h4>
          <p className="text-amber-800 leading-relaxed">
            Recycling capabilities vary significantly by municipality, collection contractor, and state infrastructure. For example, some cities accept plastic tubs (#5 PP) curbside, while others only take #1 PET and #2 HDPE bottles. Always check your local city council portal for localized curbside regulations.
          </p>
        </div>
      </div>
    </div>
  )
}
