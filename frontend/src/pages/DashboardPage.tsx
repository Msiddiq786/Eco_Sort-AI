import { useEffect, useState } from 'react'
import { LayoutDashboard, TrendingUp, Image as ImageIcon, Percent } from 'lucide-react'
import { getStats } from '@/services/api'
import type { StatsResponse } from '@/types'
import StatsCard from '@/components/dashboard/StatsCard'
import CategoryChart from '@/components/dashboard/CategoryChart'
import RecentTable from '@/components/dashboard/RecentTable'

export default function DashboardPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setError('Could not load statistics. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <LayoutDashboard className="h-6 w-6 text-primary" />
          Dashboard
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Overview of all waste classification activity.
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Stats cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatsCard
          title="Total Classified"
          value={loading ? '—' : String(stats?.total_predictions ?? 0)}
          icon={<ImageIcon className="h-5 w-5 text-primary" />}
          description="images processed"
        />
        <StatsCard
          title="Categories"
          value={loading ? '—' : String(stats?.category_distribution.length ?? 0)}
          icon={<TrendingUp className="h-5 w-5 text-primary" />}
          description="distinct waste types seen"
        />
        <StatsCard
          title="Avg. Confidence"
          value={loading ? '—' : `${((stats?.average_confidence ?? 0) * 100).toFixed(1)}%`}
          icon={<Percent className="h-5 w-5 text-primary" />}
          description="ensemble confidence"
        />
      </div>

      {/* Chart + recent table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CategoryChart distribution={stats?.category_distribution ?? []} loading={loading} />
        <RecentTable />
      </div>
    </div>
  )
}
