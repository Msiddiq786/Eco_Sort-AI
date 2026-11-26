import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

interface PageHeaderProps {
  title: string
  subtitle: string
  icon?: LucideIcon
  eyebrow?: string
  eyebrowIcon?: LucideIcon
  badge?: string
  actions?: ReactNode
}

export default function PageHeader({
  title,
  subtitle,
  icon: Icon,
  eyebrow,
  eyebrowIcon: EyebrowIcon,
  badge,
  actions,
}: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="min-w-0 flex-1">
        {(eyebrow || badge) && (
          <div className="flex flex-wrap items-center gap-2 mb-1.5">
            {eyebrow && (
              <span className="badge-primary">
                {EyebrowIcon && <EyebrowIcon className="h-3 w-3" />}
                {eyebrow}
              </span>
            )}
            {badge && (
              <span className="badge-lime uppercase tracking-wider text-[10px]">
                {badge}
              </span>
            )}
          </div>
        )}
        <h1 className="page-title">
          {Icon && (
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-ecosort-forest text-ecosort-lime shadow-sm shrink-0">
              <Icon className="h-5 w-5" />
            </span>
          )}
          <span className="truncate">{title}</span>
        </h1>
        <p className="page-subtitle">{subtitle}</p>
      </div>

      {actions && (
        <div className="flex flex-wrap items-center gap-2.5 shrink-0 self-start md:self-center">
          {actions}
        </div>
      )}
    </header>
  )
}
