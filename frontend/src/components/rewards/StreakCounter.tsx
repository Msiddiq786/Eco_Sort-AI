import { Flame, ShieldCheck } from 'lucide-react'

interface StreakProps {
  streakDays: number
  lastActiveDate: string | null
}

export default function StreakCounter({ streakDays, lastActiveDate }: StreakProps) {
  const today = new Date().toISOString().split('T')[0]
  const isActiveToday = lastActiveDate === today

  return (
    <div className="rounded-xl border border-amber-200/80 bg-gradient-to-br from-amber-50 to-orange-50/50 p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 text-white shadow-md shadow-orange-500/20">
            <Flame className="h-7 w-7 animate-pulse text-amber-100" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading text-2xl font-extrabold text-amber-950">
                {streakDays} Day{streakDays === 1 ? '' : 's'}
              </span>
              <span className="rounded-full bg-amber-200/70 px-2 py-0.5 text-[10px] font-bold text-amber-900">
                ACTIVE STREAK
              </span>
            </div>
            <p className="text-xs text-amber-800">
              {isActiveToday
                ? 'Great job! You logged recycling activity today.'
                : 'Log a confirmed sorting action today to keep your streak burning!'}
            </p>
          </div>
        </div>

        <div className="hidden sm:flex flex-col items-end text-xs text-amber-900/80">
          <div className="flex items-center gap-1 font-semibold">
            <ShieldCheck className="h-4 w-4 text-emerald-600" /> Habit Protected
          </div>
          <span className="text-[11px] text-amber-700">Streak resets if missed &gt; 24h</span>
        </div>
      </div>
    </div>
  )
}
