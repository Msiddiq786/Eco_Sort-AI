import { useState } from 'react'
import { BookOpen, HelpCircle, CheckCircle2, AlertOctagon, Layers } from 'lucide-react'
import { WASTE_CLASSES } from '@/types'
import CategoryInfoCard from '@/components/learning/CategoryInfoCard'
import DosAndDonts from '@/components/learning/DosAndDonts'
import CommonMistakes from '@/components/learning/CommonMistakes'
import RecyclingQuiz from '@/components/learning/RecyclingQuiz'

export default function LearningPage() {
  const [activeTab, setActiveTab] = useState<'categories' | 'rules' | 'mistakes' | 'quiz'>('categories')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-primary" />
          Eco Learning Center
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Master waste segregation rules, avoid contamination, and test your recycling expertise.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border space-x-2 overflow-x-auto pb-1">
        <button
          onClick={() => setActiveTab('categories')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors border-b-2 whitespace-nowrap ${
            activeTab === 'categories'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          <Layers className="h-4 w-4" /> Waste Categories & Lifecycle
        </button>

        <button
          onClick={() => setActiveTab('rules')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors border-b-2 whitespace-nowrap ${
            activeTab === 'rules'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          <CheckCircle2 className="h-4 w-4" /> Do's & Don'ts
        </button>

        <button
          onClick={() => setActiveTab('mistakes')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors border-b-2 whitespace-nowrap ${
            activeTab === 'mistakes'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          <AlertOctagon className="h-4 w-4" /> Common Mistakes
        </button>

        <button
          onClick={() => setActiveTab('quiz')}
          className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors border-b-2 whitespace-nowrap ${
            activeTab === 'quiz'
              ? 'border-primary text-primary bg-primary/5'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          <HelpCircle className="h-4 w-4" /> Interactive Quiz
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === 'categories' && (
        <div className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Explore accepted materials, degradation timelines, and preparation guidelines across all six supported recycling streams:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {WASTE_CLASSES.map(category => (
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
