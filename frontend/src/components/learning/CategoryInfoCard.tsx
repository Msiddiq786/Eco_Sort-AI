import { useState } from 'react'
import type { WasteCategory } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib'
import {
  Package,
  Wine,
  Hammer,
  FileText,
  Milk,
  Trash2,
  ChevronDown,
  ChevronUp,
  Check,
  AlertTriangle,
  RotateCcw,
} from 'lucide-react'

interface CategoryDetails {
  category: WasteCategory
  icon: typeof Package
  title: string
  binColor: string
  decompositionYears: string
  recyclabilityRate: string
  materialsIncluded: string[]
  materialsExcluded: string[]
  lifecycleSummary: string
  preparationTip: string
}

export const CATEGORY_DETAILS: Record<WasteCategory, CategoryDetails> = {
  cardboard: {
    category: 'cardboard',
    icon: Package,
    title: 'Cardboard & Paperboard',
    binColor: 'Blue / Yellow Bins',
    decompositionYears: '2 - 3 months',
    recyclabilityRate: 'Up to 7 times',
    materialsIncluded: [
      'Corrugated shipping boxes',
      'Cereal & dry food boxes (liners removed)',
      'Shoe boxes & tissue boxes',
      'Cardboard egg cartons',
    ],
    materialsExcluded: [
      'Greasy pizza boxes with cheese/oil',
      'Wax-coated frozen food boxes',
      'Cardboard soiled with paint or motor oil',
    ],
    lifecycleSummary:
      'Cardboard fibers can be repulped into new boxes up to 7 times before the cellulose fibers degrade and become too short.',
    preparationTip: 'Break down and flatten boxes completely to save space and remove plastic tape/bubble wrap.',
  },
  glass: {
    category: 'glass',
    icon: Wine,
    title: 'Container Glass',
    binColor: 'Teal / Green Bins',
    decompositionYears: '1,000,000+ years (does not decay)',
    recyclabilityRate: '100% infinitely recyclable',
    materialsIncluded: [
      'Clear, green, and amber glass bottles',
      'Food condiment jars & jam jars',
      'Beverage soda & beer bottles',
    ],
    materialsExcluded: [
      'Window panes & mirrors',
      'Ceramic coffee mugs & pyrex cookware',
      'Drinking glasses & lightbulbs',
    ],
    lifecycleSummary:
      'Glass is melted down at 1,500°C into cullet and remade into brand new bottles in 30 days without loss of purity or quality.',
    preparationTip: 'Empty liquid, rinse quickly, and separate metal lids so sorting magnets can extract both materials.',
  },
  metal: {
    category: 'metal',
    icon: Hammer,
    title: 'Aluminum & Steel Cans',
    binColor: 'Grey / Blue Bins',
    decompositionYears: '50 - 500 years',
    recyclabilityRate: '100% infinitely recyclable',
    materialsIncluded: [
      'Aluminum soda & seltzer cans',
      'Steel & tin soup / vegetable cans',
      'Clean aluminum baking foil & pie pans',
      'Empty non-hazardous aerosol cans',
    ],
    materialsExcluded: [
      'Paint cans containing wet chemical residue',
      'Propane tanks & butane canisters',
      'Car batteries & power tool scrap',
    ],
    lifecycleSummary:
      'Recycling aluminum saves 95% of the energy needed to produce virgin metal from bauxite ore. A recycled can returns to shelves in 60 days.',
    preparationTip: 'Rinse to remove food odors and pests. Do not crush cans if your city uses optical sorting scanners.',
  },
  paper: {
    category: 'paper',
    icon: FileText,
    title: 'Clean Paper & Documents',
    binColor: 'Blue Bins',
    decompositionYears: '2 - 6 weeks',
    recyclabilityRate: '5 - 7 cycles',
    materialsIncluded: [
      'Office paper, notebook sheets & printer paper',
      'Newspapers, magazines & brochures',
      'Clean envelopes (clear plastic windows are okay)',
      'Paper bags & phone books',
    ],
    materialsExcluded: [
      'Thermal receipts (contain BPA chemical coat)',
      'Wax paper & parchment paper',
      'Used paper towels, napkins & tissues',
      'Laminated paper sheets',
    ],
    lifecycleSummary:
      'Recycling 1 ton of paper preserves 17 mature trees, 26,000 liters of water, and 4,000 kWh of electric power.',
    preparationTip: 'Keep completely dry! Wet paper fibers bond prematurely and mold, causing entire recycling bales to be rejected.',
  },
  plastic: {
    category: 'plastic',
    icon: Milk,
    title: 'Rigid Plastics (#1, #2, #5)',
    binColor: 'Yellow / Orange Bins',
    decompositionYears: '450 - 1,000 years',
    recyclabilityRate: '1 - 2 cycles into textiles/lumber',
    materialsIncluded: [
      'PET #1: Water & soda bottles, clear salad containers',
      'HDPE #2: Milk jugs, shampoo & detergent bottles',
      'PP #5: Yogurt tubs, syrup bottles, medicine containers',
    ],
    materialsExcluded: [
      'Soft plastic shopping bags & shrink wrap',
      'Polystyrene / Styrofoam cups (#6)',
      'Black plastic food trays (cannot be seen by optical scanners)',
      'Plastic straws & disposable cutlery',
    ],
    lifecycleSummary:
      'Sorted plastics are shredded into flakes, washed, and pelletized into recycled resin for fleece jackets, park benches, and new bottles.',
    preparationTip: 'Empty all liquids, replace caps tightly onto bottles, and never bag recyclables in black trash sacks.',
  },
  trash: {
    category: 'trash',
    icon: Trash2,
    title: 'Residual Non-Recyclable Waste',
    binColor: 'Black / Dark Grey Bins',
    decompositionYears: 'Variable / Indefinite',
    recyclabilityRate: 'Non-recyclable (Landfill / WTE)',
    materialsIncluded: [
      'Multi-layer chip bags & snack wrappers',
      'Used hygiene wipes, diapers & dental floss',
      'Broken ceramics & porcelain dishware',
      'Greasy pizza box bottoms & food-soiled packaging',
    ],
    materialsExcluded: [
      'Hazardous household chemicals',
      'Lithium & alkaline batteries',
      'Fluorescent tubes (mercury risk)',
      'Electronics (e-waste)',
    ],
    lifecycleSummary:
      'Diverted to municipal sanitary landfills or high-efficiency Waste-to-Energy (WTE) incineration plants that generate electricity.',
    preparationTip: 'Minimize generation through bulk purchasing, reusable bags, and backyard composting of organic vegetable scraps.',
  },
}

export default function CategoryInfoCard({ category }: { category: WasteCategory }) {
  const [expanded, setExpanded] = useState(false)
  const details = CATEGORY_DETAILS[category]
  const color = CATEGORY_COLORS[category]
  const Icon = details.icon

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm transition-all hover:shadow-md">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className="flex h-11 w-11 items-center justify-center rounded-xl text-white shadow-sm"
            style={{ background: color }}
          >
            <Icon className="h-6 w-6" />
          </div>
          <div>
            <h3 className="font-heading text-lg font-bold text-slate-800">
              {capitalize(category)}
            </h3>
            <p className="text-xs text-muted-foreground">{details.title}</p>
          </div>
        </div>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
          {details.binColor}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg bg-muted/40 p-2.5">
          <span className="text-muted-foreground block text-[11px]">Recyclability</span>
          <span className="font-semibold text-slate-800">{details.recyclabilityRate}</span>
        </div>
        <div className="rounded-lg bg-muted/40 p-2.5">
          <span className="text-muted-foreground block text-[11px]">Decomposition</span>
          <span className="font-semibold text-slate-800">{details.decompositionYears}</span>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-600 leading-relaxed">
        {details.lifecycleSummary}
      </p>

      {expanded && (
        <div className="mt-4 space-y-3 pt-3 border-t border-slate-100 text-xs">
          <div>
            <h4 className="font-semibold text-emerald-800 flex items-center gap-1.5 mb-1.5">
              <Check className="h-4 w-4 text-emerald-600" /> Accepted Materials
            </h4>
            <ul className="space-y-1 pl-5 list-disc text-slate-600">
              {details.materialsIncluded.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-rose-800 flex items-center gap-1.5 mb-1.5">
              <AlertTriangle className="h-4 w-4 text-rose-500" /> Not Accepted (Contaminants)
            </h4>
            <ul className="space-y-1 pl-5 list-disc text-slate-600">
              {details.materialsExcluded.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-2.5">
            <span className="font-semibold text-emerald-900 block flex items-center gap-1">
              <RotateCcw className="h-3.5 w-3.5 text-emerald-700" /> Preparation Tip:
            </span>
            <p className="text-emerald-800 text-[11px] mt-0.5">{details.preparationTip}</p>
          </div>
        </div>
      )}

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-4 w-full flex items-center justify-center gap-1 text-xs font-semibold text-emerald-700 hover:text-emerald-800 py-1.5 rounded-lg hover:bg-emerald-50 transition-colors"
      >
        {expanded ? (
          <>Less details <ChevronUp className="h-3.5 w-3.5" /></>
        ) : (
          <>Recycling instructions & preparation <ChevronDown className="h-3.5 w-3.5" /></>
        )}
      </button>
    </div>
  )
}
