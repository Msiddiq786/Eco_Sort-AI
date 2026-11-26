import type { ReactNode } from 'react'

interface StatsCardProps {
  title: string
  value: string
  icon: ReactNode
  description?: string
}

export default function StatsCard({ title, value, icon, description }: StatsCardProps) {
  return (
    <div className="card-eco p-5 space-y-2.5 hover:shadow-eco transition-shadow">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {title}
        </p>
        <div className="text-ecosort-emerald">{icon}</div>
      </div>
      <p className="text-3xl font-heading font-extrabold text-slate-900">{value}</p>
      {description && <p className="text-xs text-slate-500">{description}</p>}
    </div>
  )
}
