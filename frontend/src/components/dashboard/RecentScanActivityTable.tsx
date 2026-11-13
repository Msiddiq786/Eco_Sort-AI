import { useState, useMemo } from 'react'
import type { HistoryItem } from '@/types'
import { capitalize, formatPct, CATEGORY_COLORS } from '@/lib/utils'
import {
  Search,
  Eye,
  Download,
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Layers,
} from 'lucide-react'
import PredictionDetailModal from '@/components/history/PredictionDetailModal'

interface RecentScanActivityTableProps {
  items: HistoryItem[]
  loading?: boolean
  onExportCsv: () => void
}

const PAGE_SIZE = 10

export default function RecentScanActivityTable({
  items,
  loading = false,
  onExportCsv,
}: RecentScanActivityTableProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)

  // Filter items
  const filtered = useMemo(() => {
    return items.filter(item => {
      if (categoryFilter !== 'all' && item.predicted_category !== categoryFilter) {
        return false
      }
      if (searchTerm) {
        const q = searchTerm.toLowerCase()
        const matchName = (item.filename || '').toLowerCase().includes(q)
        const matchOrig = (item.original_filename || '').toLowerCase().includes(q)
        const matchCat = (item.predicted_category || '').toLowerCase().includes(q)
        return matchName || matchOrig || matchCat
      }
      return true
    })
  }, [items, searchTerm, categoryFilter])

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * PAGE_SIZE
    return filtered.slice(start, start + PAGE_SIZE)
  }, [filtered, currentPage])

  // Category counts for filter
  const categoryOptions = useMemo(() => {
    const set = new Set<string>()
    items.forEach(i => {
      if (i.predicted_category) set.add(i.predicted_category)
    })
    return ['all', ...Array.from(set).sort()]
  }, [items])

  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-4">
      {/* Table Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-600" />
            <h3 className="font-heading font-bold text-base text-slate-900">
              Recent Scan Activity & Dual-Model Audit Log
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Audit trail of individual classifications, ViT/CLIP consensus votes, and ensemble confidences
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onExportCsv}
            disabled={items.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5 text-emerald-600" />
            <span>Export Table CSV</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={e => {
              setSearchTerm(e.target.value)
              setCurrentPage(1)
            }}
            placeholder="Search filenames or categories…"
            className="w-full pl-9 pr-3 py-1.5 rounded-xl border border-slate-200 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          />
        </div>

        <select
          value={categoryFilter}
          onChange={e => {
            setCategoryFilter(e.target.value)
            setCurrentPage(1)
          }}
          className="px-3 py-1.5 rounded-xl border border-slate-200 text-xs bg-white text-slate-700 cursor-pointer focus:ring-1 focus:ring-emerald-500 focus:outline-none"
        >
          {categoryOptions.map(cat => (
            <option key={cat} value={cat}>
              {cat === 'all' ? 'All Scanned Classes' : capitalize(cat)}
            </option>
          ))}
        </select>
      </div>

      {/* Table Element */}
      <div className="overflow-x-auto rounded-xl border border-slate-200/80">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50/80 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200/70 text-[10px]">
            <tr>
              <th className="py-2.5 px-3.5">ID</th>
              <th className="py-2.5 px-3">Material File</th>
              <th className="py-2.5 px-3">Predicted Category</th>
              <th className="py-2.5 px-3 text-right">Confidence</th>
              <th className="py-2.5 px-3">ViT Vote</th>
              <th className="py-2.5 px-3">CLIP Vote</th>
              <th className="py-2.5 px-3 text-center">Agreement</th>
              <th className="py-2.5 px-3">Timestamp</th>
              <th className="py-2.5 px-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {loading ? (
              <tr>
                <td colSpan={9} className="py-10 text-center text-slate-400">
                  Loading recent classification activity…
                </td>
              </tr>
            ) : paginatedItems.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-10 text-center text-slate-400">
                  No scan records match the current filters.
                </td>
              </tr>
            ) : (
              paginatedItems.map(item => {
                const color = CATEGORY_COLORS[item.predicted_category] || '#64748B'
                const isAgree =
                  item.vit_prediction &&
                  item.clip_prediction &&
                  item.vit_prediction.toLowerCase() === item.clip_prediction.toLowerCase()
                const isOrganicBypass =
                  (item.predicted_category || '').toLowerCase() === 'organic'

                const confPercent = item.confidence * 100
                const confColor =
                  confPercent >= 75
                    ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
                    : confPercent >= 50
                    ? 'text-blue-700 bg-blue-50 border-blue-200'
                    : 'text-amber-700 bg-amber-50 border-amber-200'

                const dateFormatted = new Date(item.created_at).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })

                const displayName =
                  item.original_filename || item.filename || `Scan #${item.id}`

                return (
                  <tr key={item.id} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-2.5 px-3.5 font-mono text-[11px] text-slate-400">
                      #{item.id}
                    </td>
                    <td className="py-2.5 px-3 max-w-[170px] truncate font-medium text-slate-800" title={displayName}>
                      {displayName}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="h-2 w-2 rounded-full shrink-0"
                          style={{ background: color }}
                        />
                        <span className="font-semibold text-slate-800">
                          {capitalize(item.predicted_category)}
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={`inline-block font-mono font-bold px-2 py-0.5 rounded-md border text-[11px] ${confColor}`}>
                        {formatPct(item.confidence)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 truncate max-w-[110px]" title={item.vit_prediction}>
                      {capitalize(item.vit_prediction || '—')}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 truncate max-w-[110px]" title={item.clip_prediction}>
                      {capitalize(item.clip_prediction || '—')}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {isAgree ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                          <CheckCircle2 className="h-3 w-3" /> Consensus
                        </span>
                      ) : isOrganicBypass ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-lime-100 text-lime-800 border border-lime-200">
                          Organic Bypass
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-800 border border-amber-200">
                          <AlertCircle className="h-3 w-3" /> Split (0.68)
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-[11px] text-slate-500 whitespace-nowrap">
                      {dateFormatted}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        onClick={() => setSelectedItem(item)}
                        className="p-1 rounded-lg text-slate-500 hover:text-emerald-700 hover:bg-emerald-50 transition"
                        title="View complete dual-model probability details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 pt-1">
        <span>
          Showing{' '}
          <strong>
            {filtered.length > 0 ? (currentPage - 1) * PAGE_SIZE + 1 : 0}
          </strong>{' '}
          to{' '}
          <strong>
            {Math.min(currentPage * PAGE_SIZE, filtered.length)}
          </strong>{' '}
          of <strong>{filtered.length}</strong> filtered records
        </span>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-2 text-xs font-medium">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40 transition"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Details Modal */}
      <PredictionDetailModal
        item={selectedItem}
        onClose={() => setSelectedItem(null)}
      />
    </div>
  )
}
