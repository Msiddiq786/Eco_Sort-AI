import { useState } from 'react'
import { Award, PlusCircle, History, Sparkles, CheckCircle2 } from 'lucide-react'
import { getRewardsState } from '@/services/ecoStorage'
import type { UserRewardState } from '@/types'
import { CATEGORY_COLORS } from '@/lib'
import StreakCounter from '@/components/rewards/StreakCounter'
import LevelProgress from '@/components/rewards/LevelProgress'
import DailyChallenges from '@/components/rewards/DailyChallenges'
import BadgesGrid from '@/components/rewards/BadgesGrid'
import LogActivityModal from '@/components/rewards/LogActivityModal'

export default function RewardsPage() {
  const [rewardState, setRewardState] = useState<UserRewardState>(getRewardsState)
  const [modalOpen, setModalOpen] = useState(false)

  const reload = () => {
    setRewardState(getRewardsState())
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Award className="h-6 w-6 text-primary" />
            Eco Points & Rewards
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Earn verified XP for taking genuine recycling actions, maintain your daily streak, and unlock sustainability badges.
          </p>
        </div>

        <button
          onClick={() => setModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-semibold shadow-sm transition-colors self-start sm:self-auto"
        >
          <PlusCircle className="h-4 w-4" /> Log Recycling Action
        </button>
      </div>

      {/* Top Banner Stats: Streak & Level */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StreakCounter
          streakDays={rewardState.streakDays}
          lastActiveDate={rewardState.lastActiveDate}
        />
        <LevelProgress points={rewardState.points} />
      </div>

      {/* Daily Challenges */}
      <DailyChallenges onActivityLogged={reload} />

      {/* Badges Grid */}
      <BadgesGrid unlockedBadgeIds={rewardState.unlockedBadgeIds} />

      {/* Confirmed Activities Log */}
      <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <History className="h-5 w-5 text-emerald-600" />
            <h3 className="font-heading font-bold text-base text-slate-800">
              Verified Recycling Activity Feed
            </h3>
          </div>
          <span className="text-xs text-muted-foreground">
            {rewardState.activities.length} logged actions
          </span>
        </div>

        <div className="divide-y divide-slate-100">
          {rewardState.activities.map(act => {
            const color = act.category !== 'general' ? CATEGORY_COLORS[act.category] ?? '#16A34A' : '#10B981'
            return (
              <div key={act.id} className="py-3 flex items-start justify-between gap-3 text-xs sm:text-sm">
                <div className="flex items-start gap-2.5">
                  <div
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white font-bold text-xs mt-0.5 shadow-sm"
                    style={{ background: color }}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-900">{act.title}</p>
                    {act.description && (
                      <p className="text-xs text-slate-500 mt-0.5">{act.description}</p>
                    )}
                    <span className="text-[11px] text-muted-foreground mt-1 inline-block">
                      {new Date(act.timestamp).toLocaleDateString()} at{' '}
                      {new Date(act.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>

                <div className="shrink-0 text-right">
                  <span className="inline-flex items-center gap-1 font-bold text-xs px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800">
                    <Sparkles className="h-3 w-3" /> +{act.points} pts
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <LogActivityModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={reload}
      />
    </div>
  )
}
