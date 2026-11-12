import { useState } from 'react'
import { Info, ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react'

export default function MethodologyAccordion() {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
      <div
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <Info className="h-5 w-5 text-emerald-600" />
          <div>
            <h4 className="font-heading font-bold text-sm text-slate-800">
              Calculation Methodology & Scientific Assumptions
            </h4>
            <p className="text-xs text-muted-foreground">
              How EcoSort AI computes environmental impact and greenhouse gas avoidance
            </p>
          </div>
        </div>

        <button className="text-xs font-semibold text-emerald-700 flex items-center gap-1 hover:text-emerald-800">
          {open ? (
            <>Collapse Details <ChevronUp className="h-4 w-4" /></>
          ) : (
            <>View Assumptions & Sources <ChevronDown className="h-4 w-4" /></>
          )}
        </button>
      </div>

      {open && (
        <div className="pt-3 border-t border-slate-100 text-xs text-slate-600 space-y-3 leading-relaxed">
          <div className="rounded-lg bg-emerald-50/60 p-3 border border-emerald-100 flex items-start gap-2 text-emerald-950">
            <ShieldCheck className="h-4 w-4 text-emerald-700 shrink-0 mt-0.5" />
            <p>
              <strong>Commitment to Scientific Integrity:</strong> EcoSort AI never generates arbitrary or fabricated savings figures. All impact calculations are based on conservative, empirical Lifecycle Analysis (LCA) conversion coefficients from accredited environmental research agencies.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-1.5">
              <h5 className="font-semibold text-slate-800">Greenhouse Gas (GHG) Offsets:</h5>
              <ul className="list-disc pl-4 space-y-1 text-slate-600">
                <li><strong>Metal (Aluminum/Steel):</strong> 6.2 kg CO₂e saved per kg recycled (bauxite refining avoidance).</li>
                <li><strong>Plastics (PET/HDPE):</strong> 1.4 kg CO₂e saved per kg recycled (petrochemical cracker avoidance).</li>
                <li><strong>Paper & Cardboard:</strong> 0.95 kg CO₂e saved per kg recycled (pulping energy reduction).</li>
                <li><strong>Glass Containers:</strong> 0.31 kg CO₂e saved per kg recycled (furnace energy reduction).</li>
              </ul>
            </div>

            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-1.5">
              <h5 className="font-semibold text-slate-800">Resource Conservation Factors:</h5>
              <ul className="list-disc pl-4 space-y-1 text-slate-600">
                <li><strong>Water:</strong> 26 L saved per kg paper/cardboard; 18 L per kg plastic; 12 L per kg metal.</li>
                <li><strong>Trees Saved:</strong> ~17 mature trees preserved per 1,000 kg (1 metric ton) of recycled paper (0.017 trees/kg).</li>
                <li><strong>Landfill Space:</strong> Based on standard compacted municipal density tables (0.0012 to 0.0028 m³/kg).</li>
              </ul>
            </div>
          </div>

          <div className="pt-2 text-[11px] text-muted-foreground">
            <strong>Primary Citations:</strong> US Environmental Protection Agency (EPA) <em>Waste Reduction Model (WARM v15, 2023)</em>, UK Department for Environment, Food & Rural Affairs (DEFRA) <em>Greenhouse Gas Conversion Factors</em>, and the <em>Bureau of International Recycling (BIR)</em>. Actual localized offsets depend on local electricity grid energy mixes and municipal transport radiuses.
          </div>
        </div>
      )}
    </div>
  )
}
