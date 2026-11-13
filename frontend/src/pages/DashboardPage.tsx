import { useEffect, useState, useMemo, useCallback } from 'react'
import {
  LayoutDashboard,
  Sparkles,
  Download,
  AlertCircle,
} from 'lucide-react'
import { getAllHistory, getStats } from '@/services/api'
import type {
  HistoryItem,
  StatsResponse,
  AnalyticsDateFilter,
} from '@/types'
import { CATEGORY_COLORS } from '@/lib/utils'
import DateRangeFilter from '@/components/dashboard/DateRangeFilter'
import AnalyticsSummaryCards, {
  type SummaryMetrics,
} from '@/components/dashboard/AnalyticsSummaryCards'
import CategoryDistributionChart, {
  type CategoryDataPoint,
} from '@/components/dashboard/CategoryDistributionChart'
import ScanActivityTrends from '@/components/dashboard/ScanActivityTrends'
import RecyclingInsightsSection from '@/components/dashboard/RecyclingInsightsSection'
import RecentScanActivityTable from '@/components/dashboard/RecentScanActivityTable'

const ALL_7_CATEGORIES = [
  'cardboard',
  'glass',
  'metal',
  'paper',
  'plastic',
  'organic',
  'trash',
]

const RECYCLABLE_SET = new Set(['cardboard', 'glass', 'metal', 'paper', 'plastic'])

export default function DashboardPage() {
  const [allItems, setAllItems] = useState<HistoryItem[]>([])
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Date Filtering State
  const [dateFilter, setDateFilter] = useState<AnalyticsDateFilter>('all')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')

  // Fetch full dataset from backend
  const loadData = useCallback(async (isManualRefresh = false) => {
    if (isManualRefresh) setIsRefreshing(true)
    else setLoading(true)
    setError(null)

    try {
      const [historyData, statsData] = await Promise.all([
        getAllHistory(),
        getStats().catch(() => null),
      ])
      setAllItems(historyData)
      if (statsData) setStats(statsData)
    } catch (err) {
      console.error('Failed to load analytics data:', err)
      setError(
        'Could not load prediction telemetry from backend. Ensure the FastAPI server is running on :8000.'
      )
    } finally {
      setLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Filter items by Date Range
  const filteredItems = useMemo(() => {
    if (!allItems.length) return []
    if (dateFilter === 'all') return allItems

    const now = Date.now()

    return allItems.filter(item => {
      if (!item.created_at) return true
      const itemTime = new Date(item.created_at).getTime()
      if (isNaN(itemTime)) return true

      const diffMs = now - itemTime
      const diffHours = diffMs / (1000 * 3600)

      if (dateFilter === 'today') return diffHours <= 24
      if (dateFilter === '7days') return diffHours <= 24 * 7
      if (dateFilter === '30days') return diffHours <= 24 * 30
      if (dateFilter === '90days') return diffHours <= 24 * 90

      if (dateFilter === 'custom') {
        const itemDateStr = new Date(item.created_at).toISOString().split('T')[0]
        if (customStart && itemDateStr < customStart) return false
        if (customEnd && itemDateStr > customEnd) return false
        return true
      }

      return true
    })
  }, [allItems, dateFilter, customStart, customEnd])

  // Calculate 6 Summary KPIs dynamically from filtered items
  const summaryMetrics: SummaryMetrics = useMemo(() => {
    const total = filteredItems.length
    if (total === 0) {
      return {
        totalScans: 0,
        mostDetectedCategory: null,
        recyclableCount: 0,
        recyclablePercentage: 0,
        compostableCount: 0,
        compostablePercentage: 0,
        specialDisposalCount: 0,
        specialDisposalPercentage: 0,
        averageConfidence: 0,
      }
    }

    // Counts per category
    const catCounts: Record<string, number> = {}
    let confSum = 0
    let recyclableCount = 0
    let compostableCount = 0
    let specialCount = 0

    for (const item of filteredItems) {
      const cat = (item.predicted_category || '').toLowerCase()
      catCounts[cat] = (catCounts[cat] || 0) + 1
      confSum += item.confidence || 0

      if (cat === 'organic') {
        compostableCount++
      } else if (RECYCLABLE_SET.has(cat)) {
        recyclableCount++
      } else {
        specialCount++
      }
    }

    // Most detected category
    let topCat = ''
    let topCount = 0
    for (const [cat, cnt] of Object.entries(catCounts)) {
      if (cnt > topCount) {
        topCount = cnt
        topCat = cat
      }
    }

    return {
      totalScans: total,
      mostDetectedCategory: topCat
        ? {
            category: topCat,
            count: topCount,
            percentage: (topCount / total) * 100,
          }
        : null,
      recyclableCount,
      recyclablePercentage: (recyclableCount / total) * 100,
      compostableCount,
      compostablePercentage: (compostableCount / total) * 100,
      specialDisposalCount: specialCount,
      specialDisposalPercentage: (specialCount / total) * 100,
      averageConfidence: confSum / total,
    }
  }, [filteredItems])

  // Calculate 7-Class Distribution dynamically
  const categoryDistribution: CategoryDataPoint[] = useMemo(() => {
    const total = filteredItems.length
    const counts: Record<string, number> = {}
    ALL_7_CATEGORIES.forEach(c => {
      counts[c] = 0
    })

    filteredItems.forEach(item => {
      const cat = (item.predicted_category || '').toLowerCase()
      if (counts[cat] !== undefined) {
        counts[cat]++
      } else {
        counts[cat] = (counts[cat] || 0) + 1
      }
    })

    return ALL_7_CATEGORIES.map(cat => {
      const count = counts[cat] || 0
      return {
        category: cat,
        count,
        percentage: total > 0 ? (count / total) * 100 : 0,
        color: CATEGORY_COLORS[cat] || '#64748B',
      }
    })
  }, [filteredItems])

  // Export Filtered Records as CSV
  const handleExportCsv = useCallback(() => {
    if (!filteredItems.length) return

    const headers = [
      'ID',
      'Original Filename',
      'Stored Filename',
      'Predicted Category',
      'Confidence',
      'Confidence (%)',
      'ViT Prediction',
      'CLIP Prediction',
      'Model Consensus',
      'Timestamp ISO',
    ]

    const rows = filteredItems.map(item => {
      const isAgree =
        item.vit_prediction &&
        item.clip_prediction &&
        item.vit_prediction.toLowerCase() === item.clip_prediction.toLowerCase()
      const isOrganicBypass =
        (item.predicted_category || '').toLowerCase() === 'organic'

      const consensusLabel = isAgree
        ? 'Unanimous Consensus'
        : isOrganicBypass
        ? 'Organic Bypass'
        : 'Model Disagreement'

      return [
        item.id,
        `"${(item.original_filename || '').replace(/"/g, '""')}"`,
        `"${(item.filename || '').replace(/"/g, '""')}"`,
        item.predicted_category,
        item.confidence.toFixed(4),
        (item.confidence * 100).toFixed(2) + '%',
        `"${(item.vit_prediction || '').replace(/"/g, '""')}"`,
        `"${(item.clip_prediction || '').replace(/"/g, '""')}"`,
        consensusLabel,
        `"${new Date(item.created_at).toISOString()}"`,
      ]
    })

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ecosort_analytics_${dateFilter}_${new Date().toISOString().slice(0, 10)}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [filteredItems, dateFilter])

  return (
    <div className="space-y-6">
      {/* Page Title & Live Badge */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200/80 pb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
              <Sparkles className="h-3 w-3" />
              Live Telemetry & Auditing
            </span>
            {stats && (
              <span className="text-[11px] font-medium text-slate-500">
                Lifetime Records: <strong>{stats.total_predictions}</strong>
              </span>
            )}
          </div>
          <h1 className="text-2xl font-heading font-extrabold tracking-tight text-slate-900 flex items-center gap-2.5">
            <LayoutDashboard className="h-6 w-6 text-emerald-600" />
            Advanced Analytics Dashboard
          </h1>
          <p className="text-slate-600 text-sm mt-1 max-w-2xl leading-relaxed">
            Real-world performance metrics, temporal classification trends, circular economy insights, and full audit trail from production scans.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start md:self-auto">
          <button
            onClick={handleExportCsv}
            disabled={filteredItems.length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-sm transition disabled:opacity-40"
          >
            <Download className="h-4 w-4" />
            <span>Export Analytics CSV</span>
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-800 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-bold">Backend Connection Warning</p>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Date Filter & Window Selection */}
      <DateRangeFilter
        dateFilter={dateFilter}
        onFilterChange={setDateFilter}
        customStart={customStart}
        customEnd={customEnd}
        onCustomStartChange={setCustomStart}
        onCustomEndChange={setCustomEnd}
        totalFilteredCount={filteredItems.length}
        onExportCsv={handleExportCsv}
        onRefresh={() => loadData(true)}
        isRefreshing={isRefreshing}
      />

      {/* 6 Key Analytics KPI Summary Cards */}
      <AnalyticsSummaryCards metrics={summaryMetrics} loading={loading} />

      {/* Two-Column Visual Analytics: 7-Class Distribution & Timeline Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CategoryDistributionChart
          distribution={categoryDistribution}
          totalCount={filteredItems.length}
          loading={loading}
        />
        <ScanActivityTrends items={filteredItems} loading={loading} />
      </div>

      {/* Recycling Insights: Decision Engine Statuses */}
      <RecyclingInsightsSection items={filteredItems} loading={loading} />

      {/* Real Scan Activity Table with Dual-Model Inspection & Pagination */}
      <RecentScanActivityTable
        items={filteredItems}
        loading={loading}
        onExportCsv={handleExportCsv}
      />
    </div>
  )
}
