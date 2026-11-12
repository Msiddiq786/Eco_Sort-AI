import { useState } from 'react'
import type { DailyChallenge } from '@/types'
import { getDailyChallenges, logConfirmedActivity, saveRewardsState, getRewardsState } from '@/services/ecoStorage'
import { Target, CheckCircle2, Clock, Award } from 'lucide-react'

interface DailyChallengesProps {
  onActivityLogged: () => void
}

export default function DailyChallenges({ onActivityLogged }: DailyChallengesProps) {
  const [challenges, setChallenges] = useState<DailyChallenge[]>(getDailyChallenges)

  const handleClaim = (challenge: DailyChallenge) => {
    if (challenge.completed) return

    logConfirmedActivity({
      title: `Completed Challenge: ${challenge.title}`,
      category: challenge.category ?? 'general',
      points: challenge.points,
      description: challenge.description,
    })

    const state = getRewardsState()
    if (!state.completedChallengeIds.includes(challenge.id)) {
      state.completedChallengeIds.push(challenge.id)
      saveRewardsState(state)
    }

    setChallenges(getDailyChallenges())
    onActivityLogged()
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <Target className="h-5 w-5 text-emerald-600" />
          <h3 className="font-heading font-bold text-base text-slate-800">
            Daily Recycling Challenges
          </h3>
        </div>
        <span className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground bg-slate-100 px-2 py-0.5 rounded-full">
          <Clock className="h-3 w-3" /> Resets Midnight
        </span>
      </div>

      <div className="space-y-3">
        {challenges.map(c => {
          const progressPct = Math.min(100, Math.round((c.currentCount / c.targetCount) * 100))
          return (
            <div
              key={c.id}
              className={`p-3.5 rounded-lg border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                c.completed
                  ? 'bg-emerald-50/50 border-emerald-200'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-xs sm:text-sm font-semibold text-slate-900">{c.title}</h4>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-100 text-amber-800">
                    +{c.points} XP
                  </span>
                </div>
                <p className="text-xs text-slate-600">{c.description}</p>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground pt-1">
                  <span>Progress: {c.currentCount} / {c.targetCount}</span>
                  <div className="w-20 bg-slate-100 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-emerald-600 h-full rounded-full"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                </div>
              </div>

              <div>
                {c.completed ? (
                  <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-emerald-100 text-emerald-800 text-xs font-semibold">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Completed
                  </span>
                ) : (
                  <button
                    onClick={() => handleClaim(c)}
                    className="w-full sm:w-auto px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold flex items-center justify-center gap-1 transition-colors"
                  >
                    <Award className="h-3.5 w-3.5" /> Confirm & Claim
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
