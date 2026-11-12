import { calculateLevel } from '@/services/ecoStorage'
import { Sparkles, Trophy } from 'lucide-react'

interface LevelProgressProps {
  points: number
}

export default function LevelProgress({ points }: LevelProgressProps) {
  const { level, title, progress, nextLevelPoints } = calculateLevel(points)

  return (
    <div className="rounded-xl border border-emerald-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 text-emerald-800">
            <Trophy className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-700">
                Level {level}
              </span>
              <span className="font-heading font-bold text-slate-800 text-base">
                {title}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Total Score: <strong className="text-emerald-700 font-bold">{points} Eco Points</strong>
            </p>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-semibold text-slate-600 block">
            {nextLevelPoints - points} pts to next rank
          </span>
          <span className="text-[11px] text-muted-foreground">{progress}% of current tier</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
        <div
          className="bg-gradient-to-r from-emerald-500 to-teal-600 h-full rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="mt-3 flex justify-between text-[11px] text-slate-400 font-medium">
        <span>Level {level}</span>
        <span className="flex items-center gap-1 text-emerald-700">
          <Sparkles className="h-3 w-3" /> Next: Level {Math.min(5, level + 1)}
        </span>
      </div>
    </div>
  )
}
