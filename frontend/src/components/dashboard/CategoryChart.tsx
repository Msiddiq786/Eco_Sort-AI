import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { CategoryStat } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib/utils'

interface Props {
  distribution: CategoryStat[]
  loading: boolean
}

export default function CategoryChart({ distribution, loading }: Props) {
  const data = distribution.map(d => ({
    name: capitalize(d.category),
    value: d.count,
    color: CATEGORY_COLORS[d.category] ?? '#888',
  }))

  return (
    <div className="rounded-xl border border-border/80 bg-white p-5 shadow-sm">
      <h2 className="font-heading font-semibold text-base text-slate-900 mb-4">
        Category Distribution
      </h2>
      {loading ? (
        <div className="h-56 flex items-center justify-center text-muted-foreground text-sm">
          Loading statistics…
        </div>
      ) : data.length === 0 ? (
        <div className="h-56 flex items-center justify-center text-muted-foreground text-sm">
          No classification data yet. Classify an image to see category breakdown.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              innerRadius={45}
              paddingAngle={3}
              label={({ name, percent }) =>
                `${name} ${(percent * 100).toFixed(0)}%`
              }
              labelLine={false}
            >
              {data.map(entry => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number) => [`${value} items`, 'Count']}
              contentStyle={{
                borderRadius: '8px',
                border: '1px solid #E2E8F0',
                backgroundColor: '#FFFFFF',
              }}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
