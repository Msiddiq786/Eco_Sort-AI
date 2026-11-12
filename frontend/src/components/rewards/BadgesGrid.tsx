import { ALL_BADGES } from '@/services/ecoStorage'
import { Award, Lock, CheckCircle2 } from 'lucide-react'

interface BadgesGridProps {
  unlockedBadgeIds: string[]
}

export default function BadgesGrid({ unlockedBadgeIds }: BadgesGridProps) {
  const unlockedSet = new Set(unlockedBadgeIds)

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <Award className="h-5 w-5 text-emerald-600" />
          <h3 className="font-heading font-bold text-base text-slate-800">
            Recycling Badges & Achievements
          </h3>
        </div>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">
          {unlockedSet.size} / {ALL_BADGES.length} Unlocked
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {ALL_BADGES.map(badge => {
          const isUnlocked = unlockedSet.has(badge.id)
          return (
            <div
              key={badge.id}
              className={`p-3.5 rounded-xl border flex flex-col items-center text-center transition-all ${
                isUnlocked
                  ? 'bg-gradient-to-b from-white to-emerald-50/40 border-emerald-200/90 shadow-sm'
                  : 'bg-slate-50/60 border-slate-200/70 opacity-60'
              }`}
            >
              <div
                className={`relative flex h-14 w-14 items-center justify-center rounded-2xl text-2xl mb-2.5 shadow-sm ${
                  isUnlocked
                    ? 'bg-emerald-100 border border-emerald-300'
                    : 'bg-slate-200 border border-slate-300 grayscale'
                }`}
              >
                <span>{badge.icon}</span>
                {!isUnlocked && (
                  <div className="absolute inset-0 bg-slate-900/20 backdrop-blur-[1px] rounded-2xl flex items-center justify-center">
                    <Lock className="h-4 w-4 text-slate-700" />
                  </div>
                )}
                {isUnlocked && (
                  <div className="absolute -bottom-1 -right-1 bg-emerald-600 text-white rounded-full p-0.5 shadow-sm">
                    <CheckCircle2 className="h-3 w-3" />
                  </div>
                )}
              </div>

              <h4 className="font-heading font-bold text-xs text-slate-900 leading-snug">
                {badge.name}
              </h4>
              <p className="text-[11px] text-muted-foreground mt-1 leading-tight line-clamp-2">
                {badge.description}
              </p>

              <span
                className={`mt-2 text-[10px] font-bold px-2 py-0.5 rounded-full ${
                  isUnlocked
                    ? 'bg-emerald-100 text-emerald-800'
                    : 'bg-slate-200 text-slate-600'
                }`}
              >
                {isUnlocked ? 'Unlocked' : 'Locked'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
