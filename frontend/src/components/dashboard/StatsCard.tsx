import type { ReactNode } from 'react'

interface StatsCardProps {
  title: string
  value: string
  icon: ReactNode
  description?: string
}

export default function StatsCard({ title, value, icon, description }: StatsCardProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-5 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
        {icon}
      </div>
      <p className="text-3xl font-bold">{value}</p>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
    </div>
  )
}
