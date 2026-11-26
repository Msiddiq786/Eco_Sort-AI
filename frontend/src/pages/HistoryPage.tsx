import { Fragment, useState, useEffect, useCallback, useMemo } from 'react'
import {
  History as HistoryIcon,
  Search,
  Download,
  ArrowUpDown,
  Eye,
  Calendar,
  Trash2,
  ScanSearch,
  ChevronDown,
  ChevronUp,
  Database,
} from 'lucide-react'
import {
  getHistory,
  deleteHistoryItem,
  getDetectionHistory,
  deleteDetectionHistoryItem,
} from '@/services/api'
import type {
  DetectionHistoryItem,
  HistoryItem,
  PaginatedDetectionHistory,
  PaginatedHistory,
} from '@/types'
import { V2_WASTE_CLASSES, capitalize, formatPct, CATEGORY_COLORS } from '@/lib'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
import HistorySummaryCards from '@/components/history/HistorySummaryCards'
import PredictionDetailModal from '@/components/history/PredictionDetailModal'

const PAGE_SIZE = 15

type DateFilter = 'all' | 'today' | '7days' | '30days'
type SortOrder = 'newest' | 'oldest'
type HistoryTab = 'single' | 'detections'

export default function HistoryPage() {
  const [activeTab, setActiveTab] = useState<HistoryTab>('single')
  const [data, setData] = useState<PaginatedHistory | null>(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState<string>('')
  const [dateFilter, setDateFilter] = useState<DateFilter>('all')
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)

  const [detectionData, setDetectionData] = useState<PaginatedDetectionHistory | null>(null)
  const [detPage, setDetPage] = useState(1)
  const [detLoading, setDetLoading] = useState(false)
  const [expandedScanId, setExpandedScanId] = useState<number | null>(null)

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
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

  const loadDetections = useCallback(() => {
    setDetLoading(true)
    setError(null)
    getDetectionHistory({
      page: detPage,
      page_size: PAGE_SIZE,
      search: search || undefined,
    })
      .then(setDetectionData)
      .catch(() => setError('Failed to load multi-object audit history from backend.'))
      .finally(() => setDetLoading(false))
  }, [detPage, search])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    loadDetections()
  }, [loadDetections])

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this prediction record?')) return
    await deleteHistoryItem(id)
    load()
  }

  const handleDeleteDetection = async (id: number) => {
    if (!window.confirm('Delete this multi-object detection audit record?')) return
    await deleteDetectionHistoryItem(id)
    loadDetections()
  }

  const processedItems = useMemo(() => {
    if (!data?.items) return []
    let list = [...data.items]

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

    list.sort((a, b) => {
      const tA = new Date(a.created_at).getTime()
      const tB = new Date(b.created_at).getTime()
      return sortOrder === 'newest' ? tB - tA : tA - tB
    })

    return list
  }, [data?.items, dateFilter, sortOrder])

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
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="SQLite Audit Store"
        eyebrowIcon={Database}
        icon={HistoryIcon}
        title="Classification History & Audit Log"
        subtitle="Browse, inspect, filter, correct labels, and export single-item classifications and multi-object scene audits."
        actions={
          <button
            type="button"
            onClick={exportCsv}
            disabled={!processedItems.length || activeTab !== 'single'}
            className="btn-secondary text-xs px-3.5 py-2 disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5 text-ecosort-emerald" />
            <span>Export CSV</span>
          </button>
        }
      />

      {/* Mode Tabs */}
      <div className="tab-nav">
        <button
          type="button"
          onClick={() => setActiveTab('single')}
          className={activeTab === 'single' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <HistoryIcon className="h-3.5 w-3.5" />
          <span>Single Classifications ({data?.total ?? 0})</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('detections')}
          className={activeTab === 'detections' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <ScanSearch className="h-3.5 w-3.5" />
          <span>Multi-Object Audits ({detectionData?.total ?? 0})</span>
        </button>
      </div>

      {error && (
        <StatusBanner variant="error" title="History Load Failed">
          {error}
        </StatusBanner>
      )}

      {activeTab === 'detections' ? (
        <div className="space-y-4">
          <div className="card-eco p-4 flex items-center gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                className="w-full pl-9 pr-3 py-2 rounded-xl border border-slate-200 bg-slate-50/70 text-xs sm:text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-ecosort-emerald"
                placeholder="Search multi-object audit filename…"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value)
                  setDetPage(1)
                }}
              />
            </div>
          </div>

          <div className="card-eco overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-xs sm:text-sm">
                <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600">
                  <tr>
                    <th className="text-left px-4 py-3 font-semibold">Audit # / File</th>
                    <th className="text-left px-4 py-3 font-semibold">Items</th>
                    <th className="text-left px-4 py-3 font-semibold">Material Breakdown</th>
                    <th className="text-left px-4 py-3 font-semibold">Latency</th>
                    <th className="text-left px-4 py-3 font-semibold">Timestamp</th>
                    <th className="text-right px-4 py-3 font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {detLoading ? (
                    <tr>
                      <td colSpan={6} className="text-center py-10 text-slate-400">
                        Loading multi-object audit scans…
                      </td>
                    </tr>
                  ) : !detectionData?.items?.length ? (
                    <tr>
                      <td colSpan={6} className="text-center py-10 text-slate-400">
                        No multi-object detection scans recorded yet.
                      </td>
                    </tr>
                  ) : (
                    detectionData.items.map((scan: DetectionHistoryItem) => {
                      const isExpanded = expandedScanId === scan.id
                      return (
                        <Fragment key={scan.id}>
                          <tr className="hover:bg-slate-50/60 transition-colors">
                            <td className="px-4 py-3 font-medium text-slate-800">
                              <div className="flex flex-col">
                                <span className="font-mono text-xs text-ecosort-emerald font-bold">
                                  Audit #{scan.id}
                                </span>
                                <span className="text-xs text-slate-600 truncate max-w-[200px]" title={scan.original_filename}>
                                  {scan.original_filename}
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-3 font-bold text-slate-800 whitespace-nowrap">
                              {scan.total_objects} items
                            </td>
                            <td className="px-4 py-3">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries(scan.category_counts || {}).map(([cat, count]) => (
                                  <span
                                    key={cat}
                                    className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-slate-100 text-slate-700 border border-slate-200"
                                  >
                                    {capitalize(cat)}: {count}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-4 py-3 font-mono text-[11px] text-slate-600 whitespace-nowrap">
                              {scan.inference_time_ms.toFixed(0)} ms
                            </td>
                            <td className="px-4 py-3 text-[11px] text-slate-500 whitespace-nowrap">
                              {new Date(scan.created_at).toLocaleString()}
                            </td>
                            <td className="px-4 py-3 text-right whitespace-nowrap">
                              <div className="flex items-center justify-end gap-2">
                                <button
                                  type="button"
                                  onClick={() => setExpandedScanId(isExpanded ? null : scan.id)}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium text-emerald-700 hover:bg-emerald-50"
                                  title="Inspect Item Breakdown"
                                >
                                  {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                  Inspect
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteDetection(scan.id)}
                                  className="p-1.5 rounded-lg text-rose-600 hover:bg-rose-50"
                                  title="Delete Audit Scan"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </button>
                              </div>
                            </td>
                          </tr>
                          {isExpanded && (
                            <tr key={`${scan.id}-details`} className="bg-slate-50/70">
                              <td colSpan={6} className="px-6 py-3">
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                                  {scan.objects.map((obj) => (
                                    <div
                                      key={obj.object_id}
                                      className="rounded-xl border border-slate-200 bg-white p-3 text-xs space-y-1 shadow-eco-sm"
                                    >
                                      <div className="flex items-center justify-between font-bold text-slate-800">
                                        <span>#{obj.object_id} {capitalize(obj.material_category)}</span>
                                        <span className="text-ecosort-emerald font-mono">{formatPct(obj.calibrated_confidence)}</span>
                                      </div>
                                      <div className="text-[11px] text-slate-500">
                                        ViT: {obj.vit_prediction} · CLIP: {obj.clip_prediction}
                                      </div>
                                      {obj.resin_code && (
                                        <span className="inline-block px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold text-[10px]">
                                          {obj.resin_code}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>

            {detectionData && detectionData.total > PAGE_SIZE && (
              <div className="flex justify-between items-center px-4 py-3 border-t border-slate-100 text-xs text-slate-500 bg-slate-50/40">
                <span>
                  Showing {detectionData.items.length} of {detectionData.total} audits
                </span>
                <div className="flex gap-1.5 items-center">
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg border border-slate-200 bg-white disabled:opacity-40 hover:bg-slate-50"
                    disabled={detPage <= 1}
                    onClick={() => setDetPage(p => p - 1)}
                  >
                    Prev
                  </button>
                  <span className="px-2 font-medium">
                    Page {detPage} of {Math.ceil(detectionData.total / PAGE_SIZE)}
                  </span>
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg border border-slate-200 bg-white disabled:opacity-40 hover:bg-slate-50"
                    disabled={detPage >= Math.ceil(detectionData.total / PAGE_SIZE)}
                    onClick={() => setDetPage(p => p + 1)}
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          <HistorySummaryCards
            items={processedItems}
            totalServerRecords={data?.total ?? 0}
          />

          <div className="card-eco p-4 flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                className="w-full pl-9 pr-3 py-2 rounded-xl border border-slate-200 bg-slate-50/70 text-xs sm:text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-ecosort-emerald"
                placeholder="Search original filename…"
                value={search}
                onChange={e => {
                  setSearch(e.target.value)
                  setPage(1)
                }}
              />
            </div>

            <select
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-ecosort-emerald"
              value={category}
              onChange={e => {
                setCategory(e.target.value)
                setPage(1)
              }}
            >
              <option value="">All Categories</option>
              {V2_WASTE_CLASSES.map(c => (
                <option key={c} value={c}>
                  {capitalize(c)}
                </option>
              ))}
            </select>

            <div className="flex items-center gap-1 border border-slate-200 rounded-xl p-1 bg-slate-50 text-xs">
              <Calendar className="h-3.5 w-3.5 text-slate-400 ml-1.5" />
              {(['all', 'today', '7days', '30days'] as DateFilter[]).map(df => (
                <button
                  key={df}
                  type="button"
                  onClick={() => setDateFilter(df)}
                  className={`px-2.5 py-1 rounded-lg font-medium transition-colors ${
                    dateFilter === df
                      ? 'bg-ecosort-emerald text-white shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {df === 'all' ? 'All' : df === 'today' ? '24h' : df === '7days' ? '7d' : '30d'}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => setSortOrder(s => (s === 'newest' ? 'oldest' : 'newest'))}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
              title={`Currently sorted by ${sortOrder}`}
            >
              <ArrowUpDown className="h-3.5 w-3.5 text-slate-500" />
              <span>{sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}</span>
            </button>
          </div>

          <div className="card-eco overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-xs sm:text-sm">
                <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600">
                  <tr>
                    <th className="text-left px-4 py-3 font-semibold">Category</th>
                    <th className="text-left px-4 py-3 font-semibold">Filename</th>
                    <th className="text-left px-4 py-3 font-semibold">Confidence</th>
                    <th className="text-left px-4 py-3 font-semibold">ViT Vote</th>
                    <th className="text-left px-4 py-3 font-semibold">CLIP Vote</th>
                    <th className="text-left px-4 py-3 font-semibold">Timestamp</th>
                    <th className="text-right px-4 py-3 font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {loading ? (
                    <tr>
                      <td colSpan={7} className="text-center py-10 text-slate-400">
                        Loading records from database…
                      </td>
                    </tr>
                  ) : !processedItems.length ? (
                    <tr>
                      <td colSpan={7} className="text-center py-10 text-slate-400">
                        No classification records found matching your filters.
                      </td>
                    </tr>
                  ) : (
                    processedItems.map(item => {
                      const color = CATEGORY_COLORS[item.predicted_category] ?? '#64748B'
                      const date = new Date(item.created_at).toLocaleString()
                      return (
                        <tr key={item.id} className="hover:bg-slate-50/60 transition-colors">
                          <td className="px-4 py-3 whitespace-nowrap">
                            <div className="flex items-center gap-2.5">
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

                          <td className="px-4 py-3 font-mono font-bold text-ecosort-emerald whitespace-nowrap">
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
                            <div className="flex items-center justify-end gap-1.5">
                              <button
                                type="button"
                                onClick={() => setSelectedItem(item)}
                                className="p-1.5 rounded-lg text-emerald-700 hover:bg-emerald-50 transition-colors"
                                title="Inspect Prediction Details"
                              >
                                <Eye className="h-4 w-4" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDelete(item.id)}
                                className="p-1.5 rounded-lg text-rose-600 hover:bg-rose-50 transition-colors"
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

            {data && data.total > PAGE_SIZE && (
              <div className="flex justify-between items-center px-4 py-3 border-t border-slate-100 text-xs text-slate-500 bg-slate-50/40">
                <span>
                  Showing {processedItems.length} of {data.total} records
                </span>
                <div className="flex gap-1.5 items-center">
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg border border-slate-200 bg-white disabled:opacity-40 hover:bg-slate-50"
                    disabled={page <= 1}
                    onClick={() => setPage(p => p - 1)}
                  >
                    Prev
                  </button>
                  <span className="px-2 font-medium">
                    Page {page} of {Math.ceil(data.total / PAGE_SIZE)}
                  </span>
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg border border-slate-200 bg-white disabled:opacity-40 hover:bg-slate-50"
                    disabled={page >= Math.ceil(data.total / PAGE_SIZE)}
                    onClick={() => setPage(p => p + 1)}
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>

          <PredictionDetailModal
            item={selectedItem}
            onClose={() => setSelectedItem(null)}
          />
        </>
      )}
    </div>
  )
}
