import { useState } from 'react'
import { Award, PlusCircle, History, Sparkles, CheckCircle2 } from 'lucide-react'
import { getRewardsState } from '@/services/ecoStorage'
import type { UserRewardState } from '@/types'
import { CATEGORY_COLORS } from '@/lib'
import PageHeader from '@/components/ui/PageHeader'
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
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="Sustainability Habits"
        eyebrowIcon={Award}
        badge="Local Session Tracker"
        icon={Award}
        title="Eco Points & Rewards"
        subtitle="Log confirmed recycling actions locally, maintain your daily sorting streak, and unlock educational milestones."
        actions={
          <button
            type="button"
            onClick={() => setModalOpen(true)}
            className="btn-primary text-xs px-4 py-2.5"
          >
            <PlusCircle className="h-4 w-4" />
            <span>Log Recycling Action</span>
          </button>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StreakCounter
          streakDays={rewardState.streakDays}
          lastActiveDate={rewardState.lastActiveDate}
        />
        <LevelProgress points={rewardState.points} />
      </div>

      <DailyChallenges onActivityLogged={reload} />

      <BadgesGrid unlockedBadgeIds={rewardState.unlockedBadgeIds} />

      <div className="card-eco overflow-hidden">
        <div className="section-header">
          <div className="flex items-center gap-2">
            <History className="h-4 w-4 text-ecosort-emerald" />
            <h3 className="section-title">Logged Recycling Activity Feed</h3>
          </div>
          <span className="text-xs text-slate-500">
            {rewardState.activities.length} logged actions
          </span>
        </div>

        <div className="divide-y divide-slate-100 px-5 py-2">
          {rewardState.activities.map(act => {
            const color = act.category !== 'general' ? CATEGORY_COLORS[act.category] ?? '#16A34A' : '#10B981'
            return (
              <div key={act.id} className="py-3.5 flex items-start justify-between gap-3 text-xs sm:text-sm">
                <div className="flex items-start gap-3">
                  <div
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-white font-bold text-xs mt-0.5 shadow-sm"
                    style={{ background: color }}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-900">{act.title}</p>
                    {act.description && (
                      <p className="text-xs text-slate-500 mt-0.5">{act.description}</p>
                    )}
                    <span className="text-[11px] text-slate-400 mt-1 inline-block">
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
