import type { ReactNode } from 'react'
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StatusBannerProps {
  variant?: 'error' | 'warning' | 'success' | 'info'
  title?: string
  children: ReactNode
  className?: string
}

const STYLES = {
  error: {
    container: 'border-rose-200 bg-rose-50/90 text-rose-900',
    iconColor: 'text-rose-600',
    Icon: AlertCircle,
  },
  warning: {
    container: 'border-amber-200 bg-amber-50/90 text-amber-900',
    iconColor: 'text-amber-600',
    Icon: AlertTriangle,
  },
  success: {
    container: 'border-emerald-200 bg-emerald-50/90 text-emerald-900',
    iconColor: 'text-emerald-600',
    Icon: CheckCircle2,
  },
  info: {
    container: 'border-sky-200 bg-sky-50/90 text-sky-900',
    iconColor: 'text-sky-600',
    Icon: Info,
  },
}

export default function StatusBanner({
  variant = 'info',
  title,
  children,
  className,
}: StatusBannerProps) {
  const { container, iconColor, Icon } = STYLES[variant]

  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={cn(
        'rounded-2xl border p-4 flex items-start gap-3 text-sm leading-relaxed shadow-eco-sm',
        container,
        className
      )}
    >
      <Icon className={cn('h-5 w-5 shrink-0 mt-0.5', iconColor)} />
      <div className="min-w-0 flex-1">
        {title && <p className="font-heading font-bold text-sm mb-0.5">{title}</p>}
        <div className="text-xs sm:text-sm opacity-95">{children}</div>
      </div>
    </div>
  )
}
