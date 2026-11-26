import { useEffect, useState, useMemo, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  LayoutDashboard,
  Sparkles,
  ScanLine,
  Crosshair,
} from 'lucide-react'
import { getAllHistory, getStats } from '@/services/api'
import type {
  HistoryItem,
  StatsResponse,
  AnalyticsDateFilter,
} from '@/types'
import { CATEGORY_COLORS } from '@/lib/utils'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
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

  const [dateFilter, setDateFilter] = useState<AnalyticsDateFilter>('all')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')

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
        'Could not load prediction telemetry from backend. Ensure the FastAPI server is running on port 8000.'
      )
    } finally {
      setLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

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
    <div className="space-y-6 pb-8">
      <PageHeader
        eyebrow="Live Telemetry & Auditing"
        eyebrowIcon={Sparkles}
        badge={stats ? `${stats.total_predictions} Lifetime Scans` : undefined}
        icon={LayoutDashboard}
        title="Analytics Dashboard"
        subtitle="Real-world classification telemetry, temporal trends, circular economy disposition insights, and full dual-model audit history."
        actions={
          <>
            <Link to="/classify" className="btn-primary text-xs px-3.5 py-2">
              <ScanLine className="h-4 w-4" />
              <span>Classify Item</span>
            </Link>
            <Link to="/detect" className="btn-secondary text-xs px-3.5 py-2">
              <Crosshair className="h-4 w-4 text-ecosort-emerald" />
              <span>Multi-Object Scan</span>
            </Link>
          </>
        }
      />

      {error && (
        <StatusBanner variant="error" title="Backend Connection Warning">
          {error}
        </StatusBanner>
      )}

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

      <AnalyticsSummaryCards metrics={summaryMetrics} loading={loading} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CategoryDistributionChart
          distribution={categoryDistribution}
          totalCount={filteredItems.length}
          loading={loading}
        />
        <ScanActivityTrends items={filteredItems} loading={loading} />
      </div>

      <RecyclingInsightsSection items={filteredItems} loading={loading} />

      <RecentScanActivityTable
        items={filteredItems}
        loading={loading}
        onExportCsv={handleExportCsv}
      />
    </div>
  )
}
