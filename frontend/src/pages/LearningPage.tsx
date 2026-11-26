import { useState } from 'react'
import { BookOpen, HelpCircle, CheckCircle2, AlertOctagon, Layers, GraduationCap } from 'lucide-react'
import { V2_WASTE_CLASSES } from '@/types'
import PageHeader from '@/components/ui/PageHeader'
import CategoryInfoCard from '@/components/learning/CategoryInfoCard'
import DosAndDonts from '@/components/learning/DosAndDonts'
import CommonMistakes from '@/components/learning/CommonMistakes'
import RecyclingQuiz from '@/components/learning/RecyclingQuiz'

export default function LearningPage() {
  const [activeTab, setActiveTab] = useState<'categories' | 'rules' | 'mistakes' | 'quiz'>('categories')

  return (
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="Waste Literacy & Curriculum"
        eyebrowIcon={GraduationCap}
        icon={BookOpen}
        title="Eco Learning Center"
        subtitle="Master all 7 waste segregation streams, avoid wish-cycling contamination, and test your sorting knowledge."
      />

      {/* Tabs */}
      <div className="tab-nav max-w-full overflow-x-auto scrollbar-thin">
        <button
          type="button"
          onClick={() => setActiveTab('categories')}
          className={activeTab === 'categories' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <Layers className="h-3.5 w-3.5" />
          <span>7 Waste Categories</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('rules')}
          className={activeTab === 'rules' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <CheckCircle2 className="h-3.5 w-3.5" />
          <span>Do&apos;s &amp; Don&apos;ts</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('mistakes')}
          className={activeTab === 'mistakes' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <AlertOctagon className="h-3.5 w-3.5" />
          <span>Common Mistakes</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('quiz')}
          className={activeTab === 'quiz' ? 'tab-item-active text-xs' : 'tab-item-inactive text-xs'}
        >
          <HelpCircle className="h-3.5 w-3.5" />
          <span>Interactive Quiz</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === 'categories' && (
        <div className="space-y-4">
          <p className="text-xs text-slate-500">
            Explore accepted materials, degradation timelines, and preparation guidelines across all seven EcoSort AI streams:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {V2_WASTE_CLASSES.map(category => (
              <CategoryInfoCard key={category} category={category} />
            ))}
          </div>
        </div>
      )}

      {activeTab === 'rules' && (
        <div className="space-y-4">
          <DosAndDonts />
        </div>
      )}

      {activeTab === 'mistakes' && (
        <div className="space-y-4">
          <CommonMistakes />
        </div>
      )}

      {activeTab === 'quiz' && (
        <div className="max-w-2xl mx-auto space-y-4">
          <RecyclingQuiz />
        </div>
      )}
    </div>
  )
}
