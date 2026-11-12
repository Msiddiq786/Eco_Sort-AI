import { useState } from 'react'
import type { QuizQuestion, QuizResult } from '@/types'
import { saveQuizResult, getQuizHistory } from '@/services/ecoStorage'
import { HelpCircle, CheckCircle, XCircle, RotateCcw, Award, ChevronRight } from 'lucide-react'

export const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    question: 'You ordered a pizza and the bottom of the cardboard box is soaked in grease and cheese. How should you dispose of it?',
    category: 'cardboard',
    options: [
      'Throw the entire box in the recycling bin because cardboard is recyclable',
      'Tear off the clean lid for recycling, and put the greasy bottom into the trash or compost',
      'Rinse the greasy cardboard with soap and water before recycling',
      'Burn the cardboard box in your fireplace',
    ],
    correctIndex: 1,
    explanation: 'Correct! Oil and grease permanently ruin paper pulping. The clean top lid can be recycled, while the greasy bottom belongs in trash or municipal organics compost.',
  },
  {
    id: 2,
    question: 'What should you do with plastic bottle caps before placing empty soda or water bottles in the recycling bin?',
    category: 'plastic',
    options: [
      'Remove the caps and throw them loose in the bin',
      'Tightly screw the plastic caps back onto the empty bottles',
      'Melt the caps into a ball',
      'Plastic bottles with caps cannot be recycled at all',
    ],
    correctIndex: 1,
    explanation: 'Correct! Modern recycling guidelines recommend flattening the bottle and screwing the cap tightly on. Loose caps are too small and slip through sorter screens.',
  },
  {
    id: 3,
    question: 'A drinking glass accidentally dropped and shattered on the kitchen floor. Where should the broken glass pieces go?',
    category: 'glass',
    options: [
      'In the glass recycling bin with beverage bottles and food jars',
      'Carefully wrapped in paper or cardboard, then placed in the residual trash',
      'In the metal recycling bin',
      'Melted back down in the oven',
    ],
    correctIndex: 1,
    explanation: 'Correct! Drinking glassware, mirrors, and pyrex contain lead and chemical additives with different melting points than container glass. They belong in wrapped trash.',
  },
  {
    id: 4,
    question: 'What is the most environmentally beneficial way to handle clean aluminum soda cans?',
    category: 'metal',
    options: [
      'Throw them in the landfill bin since aluminum is abundant',
      'Rinse clean and place in the metal recycling bin',
      'Bury them in the garden to enrich the soil',
      'Save them only if unprinted',
    ],
    correctIndex: 1,
    explanation: 'Correct! Aluminum is 100% infinitely recyclable. Recycling an aluminum can saves 95% of the energy needed to refine virgin aluminum from raw bauxite ore.',
  },
  {
    id: 5,
    question: 'Can paper thermal cash register receipts be recycled with normal office paper?',
    category: 'paper',
    options: [
      'Yes, all paper products can be recycled together',
      'No, receipts are coated with Bisphenol-A (BPA/BPS) chemicals that contaminate paper recycling',
      'Yes, provided they are torn into small strips',
      'Only receipts longer than 15 centimeters',
    ],
    correctIndex: 1,
    explanation: 'Correct! Thermal paper receipts contain endocrine-disrupting chemicals (BPA or BPS) that transfer into recycled paper fibers. Receipts should be discarded in trash.',
  },
  {
    id: 6,
    question: 'Why is it strictly forbidden to throw used lithium-ion or rechargeable batteries into general recycling or trash bins?',
    category: 'trash',
    options: [
      'Batteries are too heavy for collection trucks',
      'They can ignite and cause dangerous chemical fires when compacted in trucks and recycling facilities',
      'They emit radio signals that interfere with GPS',
      'They turn into gold if left in the sun',
    ],
    correctIndex: 1,
    explanation: 'Correct! Lithium-ion batteries puncture easily under compactor pressure, causing explosive thermal runaway fires. They must go to dedicated battery drop-off locations.',
  },
]

export default function RecyclingQuiz() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [selectedOption, setSelectedOption] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [score, setScore] = useState(0)
  const [quizFinished, setQuizFinished] = useState(false)
  const [history, setHistory] = useState<QuizResult[]>(getQuizHistory)

  const currentQ = QUIZ_QUESTIONS[currentIndex]

  const handleSelectOption = (idx: number) => {
    if (showExplanation) return
    setSelectedOption(idx)
    setShowExplanation(true)
    if (idx === currentQ.correctIndex) {
      setScore(s => s + 1)
    }
  }

  const handleNext = () => {
    if (currentIndex + 1 < QUIZ_QUESTIONS.length) {
      setCurrentIndex(i => i + 1)
      setSelectedOption(null)
      setShowExplanation(false)
    } else {
      const finalScore = selectedOption === currentQ.correctIndex ? score : score
      const total = QUIZ_QUESTIONS.length
      const percentage = Math.round((finalScore / total) * 100)
      const res: QuizResult = {
        score: finalScore,
        total,
        percentage,
        completedAt: new Date().toISOString(),
      }
      saveQuizResult(res)
      setHistory(getQuizHistory())
      setQuizFinished(true)
    }
  }

  const handleRestart = () => {
    setCurrentIndex(0)
    setSelectedOption(null)
    setShowExplanation(false)
    setScore(0)
    setQuizFinished(false)
  }

  return (
    <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <HelpCircle className="h-5 w-5 text-emerald-600" />
          <h3 className="font-heading font-bold text-lg text-slate-800">
            Waste Segregation Knowledge Quiz
          </h3>
        </div>
        {!quizFinished && (
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800">
            Question {currentIndex + 1} of {QUIZ_QUESTIONS.length}
          </span>
        )}
      </div>

      {!quizFinished ? (
        <div className="mt-5 space-y-5">
          {/* Progress bar */}
          <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
            <div
              className="bg-emerald-600 h-full transition-all duration-300"
              style={{ width: `${((currentIndex + 1) / QUIZ_QUESTIONS.length) * 100}%` }}
            />
          </div>

          <h4 className="text-base font-semibold text-slate-900 leading-snug">
            {currentQ.question}
          </h4>

          {/* Options */}
          <div className="space-y-2.5">
            {currentQ.options.map((option, idx) => {
              const isSelected = selectedOption === idx
              const isCorrect = idx === currentQ.correctIndex

              let btnStyle = 'border-slate-200 hover:border-emerald-500 hover:bg-emerald-50/30'
              if (showExplanation) {
                if (isCorrect) {
                  btnStyle = 'border-emerald-500 bg-emerald-50 text-emerald-950 font-medium'
                } else if (isSelected) {
                  btnStyle = 'border-rose-400 bg-rose-50 text-rose-950'
                } else {
                  btnStyle = 'border-slate-100 opacity-60'
                }
              }

              return (
                <button
                  key={idx}
                  onClick={() => handleSelectOption(idx)}
                  disabled={showExplanation}
                  className={`w-full text-left p-3.5 rounded-lg border text-xs sm:text-sm transition-all flex items-center justify-between gap-3 ${btnStyle}`}
                >
                  <span>{option}</span>
                  {showExplanation && isCorrect && (
                    <CheckCircle className="h-4 w-4 shrink-0 text-emerald-600" />
                  )}
                  {showExplanation && isSelected && !isCorrect && (
                    <XCircle className="h-4 w-4 shrink-0 text-rose-500" />
                  )}
                </button>
              )
            })}
          </div>

          {/* Explanation Callout */}
          {showExplanation && (
            <div
              className={`p-4 rounded-lg text-xs leading-relaxed border ${
                selectedOption === currentQ.correctIndex
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                  : 'bg-amber-50 border-amber-200 text-amber-900'
              }`}
            >
              <p className="font-semibold mb-1">
                {selectedOption === currentQ.correctIndex ? 'Spot on!' : 'Not quite right:'}
              </p>
              <p>{currentQ.explanation}</p>
            </div>
          )}

          {showExplanation && (
            <button
              onClick={handleNext}
              className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-semibold flex items-center justify-center gap-1.5 transition-colors"
            >
              {currentIndex + 1 < QUIZ_QUESTIONS.length ? (
                <>Next Question <ChevronRight className="h-4 w-4" /></>
              ) : (
                <>View Final Score <Award className="h-4 w-4" /></>
              )}
            </button>
          )}
        </div>
      ) : (
        /* Quiz Finished State */
        <div className="mt-6 text-center space-y-4">
          <div className="inline-flex p-4 rounded-full bg-emerald-100 text-emerald-700 mb-2">
            <Award className="h-10 w-10" />
          </div>
          <h4 className="text-xl font-heading font-bold text-slate-800">
            Quiz Completed!
          </h4>
          <p className="text-sm text-muted-foreground">
            You scored <strong className="text-emerald-700 font-bold">{score} out of {QUIZ_QUESTIONS.length}</strong> ({Math.round((score / QUIZ_QUESTIONS.length) * 100)}%)
          </p>

          <p className="text-xs text-slate-600 max-w-md mx-auto">
            {score >= 5
              ? 'Outstanding! You have mastered municipal recycling guidelines and have earned the Quiz Master award.'
              : 'Good effort! Review the waste category guidelines above and retake the quiz to sharpen your sorting knowledge.'}
          </p>

          <button
            onClick={handleRestart}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold transition-colors"
          >
            <RotateCcw className="h-4 w-4" /> Retake Quiz
          </button>

          {/* Past Attempts history */}
          {history.length > 0 && (
            <div className="mt-6 pt-4 border-t border-slate-100 text-left">
              <h5 className="text-xs font-semibold text-slate-700 mb-2">Recent Quiz Attempts</h5>
              <div className="space-y-1.5">
                {history.slice(0, 3).map((item, idx) => (
                  <div key={idx} className="flex justify-between items-center text-xs p-2 rounded bg-slate-50">
                    <span className="text-slate-600">
                      {new Date(item.completedAt).toLocaleDateString()} at {new Date(item.completedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="font-semibold text-emerald-700">
                      {item.score}/{item.total} ({item.percentage}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
