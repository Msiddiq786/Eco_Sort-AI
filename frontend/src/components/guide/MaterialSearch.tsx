import { useState, useMemo } from 'react'
import {
  Search,
  Filter,
  RotateCcw,
  Layers,
} from 'lucide-react'
import { RECYCLING_GUIDE_MATERIALS } from '@/data/recyclingGuideData'
import type { RecyclingGuideMaterial } from '@/types'
import MaterialCard from './MaterialCard'
import MaterialDetailModal from './MaterialDetailModal'

const CATEGORY_TABS = [
  { id: 'all', label: 'All Categories' },
  { id: 'cardboard', label: 'Cardboard' },
  { id: 'glass', label: 'Glass' },
  { id: 'metal', label: 'Metal' },
  { id: 'paper', label: 'Paper' },
  { id: 'plastic', label: 'Plastic' },
  { id: 'organic', label: 'Organic' },
  { id: 'trash', label: 'Trash' },
  { id: 'hazardous', label: 'Special / Hazardous' },
]

const STATUS_OPTIONS = [
  'All Statuses',
  'Widely Recyclable',
  'Conditionally Recyclable',
  'Compostable',
  'Special Disposal Required',
  'Residual Waste (Non-Recyclable)',
]

export default function MaterialSearch() {
  const [searchTerm, setSearchTerm] = useState('')
  const [activeCategory, setActiveCategory] = useState('all')
  const [activeStatus, setActiveStatus] = useState('All Statuses')
  const [selectedMaterial, setSelectedMaterial] = useState<RecyclingGuideMaterial | null>(null)

  // Filter logic
  const filteredMaterials = useMemo(() => {
    const q = searchTerm.toLowerCase().trim()

    return RECYCLING_GUIDE_MATERIALS.filter(m => {
      // Category filter
      if (activeCategory !== 'all' && m.category !== activeCategory) {
        return false
      }

      // Status filter
      if (activeStatus !== 'All Statuses' && m.recyclingPossibility !== activeStatus) {
        return false
      }

      // Search term filter (name, description, examples, disposal method)
      if (q) {
        const matchesName = m.name.toLowerCase().includes(q)
        const matchesDesc = m.description.toLowerCase().includes(q)
        const matchesExamples = m.commonExamples.some(ex => ex.toLowerCase().includes(q))
        const matchesBin = m.disposalMethod.binName.toLowerCase().includes(q)
        const matchesPrep = m.preparationInstructions.some(p => p.toLowerCase().includes(q))
        return matchesName || matchesDesc || matchesExamples || matchesBin || matchesPrep
      }

      return true
    })
  }, [searchTerm, activeCategory, activeStatus])

  const hasActiveFilters = searchTerm !== '' || activeCategory !== 'all' || activeStatus !== 'All Statuses'

  const handleClearFilters = () => {
    setSearchTerm('')
    setActiveCategory('all')
    setActiveStatus('All Statuses')
  }

  return (
    <div className="space-y-6">
      {/* Search & Filter Toolbar */}
      <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          {/* Search bar */}
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              placeholder="Search by material name, examples (e.g. pizza box, banana peel, milk jug, batteries)..."
              className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-semibold text-slate-400 hover:text-slate-600"
              >
                Clear
              </button>
            )}
          </div>

          {/* Status Dropdown */}
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-slate-400 shrink-0" />
            <select
              value={activeStatus}
              onChange={e => setActiveStatus(e.target.value)}
              className="py-2.5 px-3 rounded-xl border border-slate-200 text-sm bg-white text-slate-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition cursor-pointer"
            >
              {STATUS_OPTIONS.map(opt => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>

          {/* Reset Filters button */}
          {hasActiveFilters && (
            <button
              onClick={handleClearFilters}
              className="flex items-center justify-center gap-1.5 px-3.5 py-2.5 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </button>
          )}
        </div>

        {/* Category Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
          {CATEGORY_TABS.map(tab => {
            const count =
              tab.id === 'all'
                ? RECYCLING_GUIDE_MATERIALS.length
                : RECYCLING_GUIDE_MATERIALS.filter(m => m.category === tab.id).length

            return (
              <button
                key={tab.id}
                onClick={() => setActiveCategory(tab.id)}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium whitespace-nowrap transition-all ${
                  activeCategory === tab.id
                    ? 'bg-emerald-600 text-white shadow-sm font-semibold'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200/70 hover:text-slate-900'
                }`}
              >
                <span>{tab.label}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full ${
                    activeCategory === tab.id
                      ? 'bg-emerald-700 text-white'
                      : 'bg-slate-200/80 text-slate-600'
                  }`}
                >
                  {count}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Results Header */}
      <div className="flex items-center justify-between text-xs text-slate-500 px-1">
        <div className="flex items-center gap-1.5">
          <Layers className="h-4 w-4 text-emerald-600" />
          <span>
            Showing <strong>{filteredMaterials.length}</strong> of{' '}
            {RECYCLING_GUIDE_MATERIALS.length} verified materials
          </span>
        </div>
        {hasActiveFilters && (
          <span className="text-emerald-700 font-medium">Filters Applied</span>
        )}
      </div>

      {/* Materials Grid */}
      {filteredMaterials.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredMaterials.map(material => (
            <MaterialCard
              key={material.id}
              material={material}
              onSelect={setSelectedMaterial}
            />
          ))}
        </div>
      ) : (
        /* Empty State */
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-12 text-center space-y-3">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-50 text-amber-600">
            <Search className="h-6 w-6" />
          </div>
          <h3 className="font-heading font-bold text-base text-slate-800">
            No waste materials matched your query
          </h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
            No catalog items matched "{searchTerm}" with the selected category and status filters. Try refining your keywords or clear your filters to view all 7 waste categories.
          </p>
          <div className="pt-2">
            <button
              onClick={handleClearFilters}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 transition shadow-sm"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset All Filters
            </button>
          </div>
        </div>
      )}

      {/* Interactive Detail Modal */}
      <MaterialDetailModal
        material={selectedMaterial}
        onClose={() => setSelectedMaterial(null)}
      />
    </div>
  )
}
