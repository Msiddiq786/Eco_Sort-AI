import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  History as HistoryIcon,
  Search,
  Download,
  ArrowUpDown,
  Eye,
  Calendar,
  Trash2,
} from 'lucide-react'
import { getHistory, deleteHistoryItem } from '@/services/api'
import type { HistoryItem, PaginatedHistory } from '@/types'
import { WASTE_CLASSES, capitalize, formatPct, CATEGORY_COLORS } from '@/lib'
import HistorySummaryCards from '@/components/history/HistorySummaryCards'
import PredictionDetailModal from '@/components/history/PredictionDetailModal'

const PAGE_SIZE = 15

type DateFilter = 'all' | 'today' | '7days' | '30days'
type SortOrder = 'newest' | 'oldest'

export default function HistoryPage() {
  const [data, setData] = useState<PaginatedHistory | null>(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState<string>('')
  const [dateFilter, setDateFilter] = useState<DateFilter>('all')
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)

  const load = useCallback(() => {
    setLoading(true)
    getHistory({
      page,
      page_size: PAGE_SIZE,
      category: category || undefined,
      search: search || undefined,
    })
      .then(setData)
      .catch(() => setError('Failed to load history from backend.'))
      .finally(() => setLoading(false))
  }, [page, category, search])

  useEffect(() => {
    load()
  }, [load])

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this prediction record?')) return
    await deleteHistoryItem(id)
    load()
  }

  // Client-side date filtering & sorting on loaded items
  const processedItems = useMemo(() => {
    if (!data?.items) return []
    let list = [...data.items]

    // Date filtering
    if (dateFilter !== 'all') {
      const now = Date.now()
      list = list.filter(item => {
        const itemTime = new Date(item.created_at).getTime()
        const diffHours = (now - itemTime) / (1000 * 3600)
        if (dateFilter === 'today') return diffHours <= 24
        if (dateFilter === '7days') return diffHours <= 24 * 7
        if (dateFilter === '30days') return diffHours <= 24 * 30
        return true
      })
    }

    // Sort order
    list.sort((a, b) => {
      const tA = new Date(a.created_at).getTime()
      const tB = new Date(b.created_at).getTime()
      return sortOrder === 'newest' ? tB - tA : tA - tB
    })

    return list
  }, [data?.items, dateFilter, sortOrder])

  // CSV Export
  const exportCsv = () => {
    if (!processedItems.length) return
    const headers = ['ID', 'Filename', 'Predicted Category', 'Confidence', 'ViT Prediction', 'CLIP Prediction', 'Date']
    const rows = processedItems.map(item => [
      item.id,
      `"${item.filename.replace(/"/g, '""')}"`,
      item.predicted_category,
      (item.confidence * 100).toFixed(2) + '%',
      item.vit_prediction,
      item.clip_prediction,
      `"${new Date(item.created_at).toISOString()}"`,
    ])

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.setAttribute('href', url)
    link.setAttribute('download', `ecosort_history_${new Date().toISOString().split('T')[0]}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <HistoryIcon className="h-6 w-6 text-primary" />
            Classification History & Audit Log
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Browse, inspect, filter, and export waste classification records.
          </p>
        </div>

        <button
          onClick={exportCsv}
          disabled={!processedItems.length}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-sm transition-colors self-start sm:self-auto disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5 text-emerald-600" /> Export CSV
        </button>
      </div>

      {/* Summary Cards */}
      <HistorySummaryCards
        items={processedItems}
        totalServerRecords={data?.total ?? 0}
      />

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center gap-3 p-4 rounded-xl bg-card border border-border shadow-sm">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-input bg-background text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="Search original filename…"
            value={search}
            onChange={e => {
              setSearch(e.target.value)
              setPage(1)
            }}
          />
        </div>

        {/* Category Filter */}
        <select
          className="rounded-lg border border-input bg-background px-3 py-2 text-xs sm:text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          value={category}
          onChange={e => {
            setCategory(e.target.value)
            setPage(1)
          }}
        >
          <option value="">All Categories</option>
          {WASTE_CLASSES.map(c => (
            <option key={c} value={c}>
              {capitalize(c)}
            </option>
          ))}
        </select>

        {/* Date Filter */}
        <div className="flex items-center gap-1 border border-input rounded-lg p-1 bg-background text-xs">
          <Calendar className="h-3.5 w-3.5 text-muted-foreground ml-1.5" />
          <button
            onClick={() => setDateFilter('all')}
            className={`px-2 py-1 rounded font-medium transition-colors ${dateFilter === 'all' ? 'bg-primary text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            All
          </button>
          <button
            onClick={() => setDateFilter('today')}
            className={`px-2 py-1 rounded font-medium transition-colors ${dateFilter === 'today' ? 'bg-primary text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            24h
          </button>
          <button
            onClick={() => setDateFilter('7days')}
            className={`px-2 py-1 rounded font-medium transition-colors ${dateFilter === '7days' ? 'bg-primary text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            7d
          </button>
          <button
            onClick={() => setDateFilter('30days')}
            className={`px-2 py-1 rounded font-medium transition-colors ${dateFilter === '30days' ? 'bg-primary text-white' : 'text-slate-600 hover:text-slate-900'}`}
          >
            30d
          </button>
        </div>

        {/* Sort Toggle */}
        <button
          onClick={() => setSortOrder(s => (s === 'newest' ? 'oldest' : 'newest'))}
          className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-input bg-background text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
          title={`Currently sorted by ${sortOrder}`}
        >
          <ArrowUpDown className="h-3.5 w-3.5 text-slate-500" />
          <span>{sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}</span>
        </button>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs">
          {error}
        </div>
      )}

      {/* History Table */}
      <div className="rounded-xl border border-border bg-card overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-xs sm:text-sm">
            <thead className="bg-slate-50/80 border-b border-border text-slate-600">
              <tr>
                <th className="text-left px-4 py-3 font-semibold">Preview / Category</th>
                <th className="text-left px-4 py-3 font-semibold">Filename</th>
                <th className="text-left px-4 py-3 font-semibold">Confidence</th>
                <th className="text-left px-4 py-3 font-semibold">ViT Vote</th>
                <th className="text-left px-4 py-3 font-semibold">CLIP Vote</th>
                <th className="text-left px-4 py-3 font-semibold">Timestamp</th>
                <th className="text-right px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {loading ? (
                <tr>
                  <td colSpan={7} className="text-center py-10 text-muted-foreground">
                    Loading records from database…
                  </td>
                </tr>
              ) : !processedItems.length ? (
                <tr>
                  <td colSpan={7} className="text-center py-10 text-muted-foreground">
                    No classification records found matching your filters.
                  </td>
                </tr>
              ) : (
                processedItems.map(item => {
                  const color = CATEGORY_COLORS[item.predicted_category] ?? '#888'
                  const date = new Date(item.created_at).toLocaleString()
                  return (
                    <tr key={item.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div
                            className="flex h-7 w-7 items-center justify-center rounded-lg text-white font-bold text-xs shadow-sm"
                            style={{ background: color }}
                          >
                            {capitalize(item.predicted_category).charAt(0)}
                          </div>
                          <span className="font-semibold text-slate-800">
                            {capitalize(item.predicted_category)}
                          </span>
                        </div>
                      </td>

                      <td
                        className="px-4 py-3 font-mono text-[11px] max-w-[160px] truncate text-slate-700"
                        title={item.filename}
                      >
                        {item.filename}
                      </td>

                      <td className="px-4 py-3 font-semibold text-emerald-700 whitespace-nowrap">
                        {formatPct(item.confidence)}
                      </td>

                      <td className="px-4 py-3 text-slate-600 capitalize whitespace-nowrap">
                        {item.vit_prediction}
                      </td>

                      <td className="px-4 py-3 text-slate-600 capitalize whitespace-nowrap">
                        {item.clip_prediction}
                      </td>

                      <td className="px-4 py-3 text-slate-500 whitespace-nowrap text-[11px]">
                        {date}
                      </td>

                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => setSelectedItem(item)}
                            className="p-1 rounded text-emerald-700 hover:bg-emerald-50 transition-colors"
                            title="Inspect Prediction Details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(item.id)}
                            className="p-1 rounded text-rose-600 hover:bg-rose-50 transition-colors"
                            title="Delete Record"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data && data.total > PAGE_SIZE && (
          <div className="flex justify-between items-center px-4 py-3 border-t border-border text-xs text-muted-foreground bg-slate-50/40">
            <span>
              Showing {processedItems.length} of {data.total} records
            </span>
            <div className="flex gap-1.5 items-center">
              <button
                className="px-3 py-1 rounded-md border border-border bg-white disabled:opacity-40 hover:bg-slate-50"
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
              >
                Prev
              </button>
              <span className="px-2">
                Page {page} of {Math.ceil(data.total / PAGE_SIZE)}
              </span>
              <button
                className="px-3 py-1 rounded-md border border-border bg-white disabled:opacity-40 hover:bg-slate-50"
                disabled={page >= Math.ceil(data.total / PAGE_SIZE)}
                onClick={() => setPage(p => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Prediction Inspection Modal */}
      <PredictionDetailModal
        item={selectedItem}
        onClose={() => setSelectedItem(null)}
      />
    </div>
  )
}
