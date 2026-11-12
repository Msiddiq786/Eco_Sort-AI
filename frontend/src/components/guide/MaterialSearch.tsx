import { useState } from 'react'
import type { WasteCategory } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib'
import { Search } from 'lucide-react'

interface ItemGuide {
  name: string
  category: WasteCategory | 'hazardous' | 'dropoff'
  bin: string
  preparation: string
  upcycleTip?: string
}

export const ITEM_DATABASE: ItemGuide[] = [
  {
    name: 'Plastic Water / Soda Bottle (#1 PET)',
    category: 'plastic',
    bin: 'Plastic Recycling Bin',
    preparation: 'Empty liquid, crush bottle flat, screw cap tightly back on.',
    upcycleTip: 'Can be repurposed as a seedling greenhouse or bird feeder.',
  },
  {
    name: 'Milk Jug & Detergent Bottle (#2 HDPE)',
    category: 'plastic',
    bin: 'Plastic Recycling Bin',
    preparation: 'Rinse out remaining milk or detergent film. Keep cap on.',
    upcycleTip: 'Heavy duty plastic makes durable garden scoops or water weights.',
  },
  {
    name: 'Aluminum Beverage Can',
    category: 'metal',
    bin: 'Metal Recycling Bin',
    preparation: 'Empty liquid. Quick rinse. Do not crush if your city uses optical sorters.',
    upcycleTip: 'Infinitely recyclable back into a new can within 60 days.',
  },
  {
    name: 'Tin / Steel Food Soup Can',
    category: 'metal',
    bin: 'Metal Recycling Bin',
    preparation: 'Rinse clean. Place the metal lid inside the can to avoid sharp hazards.',
    upcycleTip: 'Clean cans make excellent pencil holders or desktop organizers.',
  },
  {
    name: 'Corrugated Shipping Box',
    category: 'cardboard',
    bin: 'Cardboard Bin',
    preparation: 'Remove plastic tape and packing peanuts. Flatten completely.',
    upcycleTip: 'Use as weed-blocking mulch sheet in garden beds.',
  },
  {
    name: 'Greasy Pizza Box Bottom',
    category: 'trash',
    bin: 'Residual Trash / Organics Bin',
    preparation: 'Tear off clean lid for cardboard recycling; put oily base in trash.',
    upcycleTip: 'If your city has municipal compost, greasy cardboard is compostable.',
  },
  {
    name: 'Wine & Beer Glass Bottle',
    category: 'glass',
    bin: 'Glass Recycling Bin',
    preparation: 'Empty liquid. Remove metal caps and corks.',
    upcycleTip: 'Glass bottles can be reused as flower vases or olive oil dispensers.',
  },
  {
    name: 'Peanut Butter Glass Jar',
    category: 'glass',
    bin: 'Glass Recycling Bin',
    preparation: 'Scrape clean with silicone spatula, quick rinse. Separate metal lid.',
    upcycleTip: 'Mason jars are perfect for dry pantry staples and bulk beans.',
  },
  {
    name: 'Newspaper & Magazines',
    category: 'paper',
    bin: 'Paper Recycling Bin',
    preparation: 'Keep dry. Remove plastic sleeve wrappers.',
  },
  {
    name: 'Office & Printer Paper',
    category: 'paper',
    bin: 'Paper Recycling Bin',
    preparation: 'Keep dry. Staples do not need to be removed.',
    upcycleTip: 'Use blank backsides of printed sheets as scratch memo paper.',
  },
  {
    name: 'Plastic Grocery Shopping Bags',
    category: 'dropoff',
    bin: 'Supermarket Plastic Film Drop-off',
    preparation: 'Never put loose in curbside bins. Drop at grocery store bins.',
    upcycleTip: 'Keep in car trunk to reuse for dog walks or small bin liners.',
  },
  {
    name: 'Takeout Coffee Cups',
    category: 'trash',
    bin: 'Residual Trash Bin',
    preparation: 'Discard in trash. Separate plastic lid into plastic recycling.',
  },
  {
    name: 'Aerosol Spray Cans (Deodorant/Hair)',
    category: 'metal',
    bin: 'Metal Bin (Must be 100% empty)',
    preparation: 'Discharge all pressure. If still hissing or containing liquid, take to hazardous waste.',
  },
  {
    name: 'Drinking Glass / Broken Mirror',
    category: 'trash',
    bin: 'Residual Trash Bin (Wrapped)',
    preparation: 'Wrap in newspaper or cardboard to protect waste collection staff.',
  },
  {
    name: 'Lithium / AA / AAA Batteries',
    category: 'hazardous',
    bin: 'Battery Recycling Kiosk',
    preparation: 'Tape battery terminals with scotch tape. Never put in trash/curbside bins.',
  },
]

export default function MaterialSearch() {
  const [query, setQuery] = useState('')
  const [selectedFilter, setSelectedFilter] = useState<string>('all')

  const filtered = ITEM_DATABASE.filter(item => {
    const matchesQuery = item.name.toLowerCase().includes(query.toLowerCase()) ||
      item.preparation.toLowerCase().includes(query.toLowerCase())
    const matchesCat = selectedFilter === 'all' || item.category === selectedFilter
    return matchesQuery && matchesCat
  })

  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-sm space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h3 className="font-heading font-bold text-lg text-slate-800">
            Household Item Sorting Directory
          </h3>
          <p className="text-xs text-muted-foreground">
            Search 15+ everyday household items for disposal guidelines and upcycling tips
          </p>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-input bg-background text-xs focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="Type 'coffee cup', 'pizza'…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Filter Pills */}
      <div className="flex flex-wrap gap-1.5 text-xs">
        {['all', 'plastic', 'glass', 'metal', 'paper', 'cardboard', 'trash', 'hazardous'].map(filter => (
          <button
            key={filter}
            onClick={() => setSelectedFilter(filter)}
            className={`px-3 py-1.5 rounded-lg capitalize font-medium transition-colors ${
              selectedFilter === filter
                ? 'bg-primary text-white shadow-xs'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {filter}
          </button>
        ))}
      </div>

      {/* Item List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pt-2">
        {filtered.map((item, idx) => {
          const isStandardWaste = item.category in CATEGORY_COLORS
          const color = isStandardWaste ? CATEGORY_COLORS[item.category] : item.category === 'hazardous' ? '#E11D48' : '#6366F1'
          return (
            <div
              key={idx}
              className="p-4 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h4 className="font-semibold text-xs sm:text-sm text-slate-900 leading-snug">
                    {item.name}
                  </h4>
                  <span
                    className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-bold text-white shadow-xs"
                    style={{ background: color }}
                  >
                    {capitalize(item.category)}
                  </span>
                </div>

                <div className="rounded-lg bg-slate-50 p-2.5 text-xs text-slate-700 mb-2 border border-slate-100">
                  <span className="font-semibold block text-[11px] text-slate-500 mb-0.5">
                    Target Bin:
                  </span>
                  <span className="font-bold text-slate-800">{item.bin}</span>
                  <p className="text-[11px] text-slate-600 mt-1 leading-snug">{item.preparation}</p>
                </div>
              </div>

              {item.upcycleTip && (
                <div className="pt-2 border-t border-slate-100 flex items-start gap-1.5 text-[11px] text-emerald-800">
                  <span className="font-bold shrink-0">💡 Upcycle:</span>
                  <span>{item.upcycleTip}</span>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
