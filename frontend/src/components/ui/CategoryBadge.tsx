import { capitalize, CATEGORY_COLORS, cn } from '@/lib/utils'

interface CategoryBadgeProps {
  category: string
  size?: 'sm' | 'md'
  className?: string
}

export default function CategoryBadge({
  category,
  size = 'md',
  className,
}: CategoryBadgeProps) {
  const normalized = (category || '').toLowerCase()
  const color = CATEGORY_COLORS[normalized] ?? '#64748B'

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-semibold border whitespace-nowrap',
        size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs',
        className
      )}
      style={{
        backgroundColor: `${color}14`,
        borderColor: `${color}38`,
        color: '#1E293B',
      }}
    >
      <span
        className="h-2 w-2 rounded-full shrink-0"
        style={{ backgroundColor: color }}
      />
      <span>{capitalize(normalized || 'unknown')}</span>
    </span>
  )
}
