import type {
  Badge,
  DailyChallenge,
  EcoActivity,
  ImpactSummary,
  QuizResult,
  UserRewardState,
  WasteCategory,
  WasteWeightLog,
} from '@/types'

const REWARDS_STORAGE_KEY = 'ecosort_rewards_state_v1'
const QUIZ_STORAGE_KEY = 'ecosort_quiz_history_v1'
const WEIGHT_LOGS_STORAGE_KEY = 'ecosort_weight_logs_v1'

// Standard badges definition
export const ALL_BADGES: Badge[] = [
  {
    id: 'first_sort',
    name: 'First Sort',
    description: 'Logged your first confirmed recycling action.',
    icon: '🌟',
    category: 'milestone',
    unlockedAt: null,
  },
  {
    id: 'plastic_pioneer',
    name: 'Plastic Pioneer',
    description: 'Cleaned and recycled 5 plastic containers.',
    icon: '🧴',
    category: 'plastic',
    unlockedAt: null,
  },
  {
    id: 'cardboard_crusher',
    name: 'Cardboard Crusher',
    description: 'Flattened and recycled 5 cardboard boxes.',
    icon: '📦',
    category: 'cardboard',
    unlockedAt: null,
  },
  {
    id: 'metal_maestro',
    name: 'Metal Maestro',
    description: 'Recycled 5 aluminum or tin cans.',
    icon: '🥫',
    category: 'metal',
    unlockedAt: null,
  },
  {
    id: 'glass_guardian',
    name: 'Glass Guardian',
    description: 'Safely separated 5 glass bottles or jars.',
    icon: '🍾',
    category: 'glass',
    unlockedAt: null,
  },
  {
    id: 'quiz_master',
    name: 'Quiz Master',
    description: 'Achieved 100% on the Waste Segregation Quiz.',
    icon: '🎓',
    category: 'education',
    unlockedAt: null,
  },
  {
    id: 'streak_3',
    name: 'Habit Builder',
    description: 'Maintained a 3-day recycling streak.',
    icon: '🔥',
    category: 'streak',
    unlockedAt: null,
  },
  {
    id: 'streak_7',
    name: 'Eco Champion',
    description: 'Maintained a 7-day recycling streak.',
    icon: '⚡',
    category: 'streak',
    unlockedAt: null,
  },
]

export const LEVEL_THRESHOLDS = [
  { level: 1, name: 'Novice Sorter', minPoints: 0, maxPoints: 99 },
  { level: 2, name: 'Eco Ally', minPoints: 100, maxPoints: 249 },
  { level: 3, name: 'Green Guardian', minPoints: 250, maxPoints: 499 },
  { level: 4, name: 'Earth Defender', minPoints: 500, maxPoints: 999 },
  { level: 5, name: 'Zero-Waste Champion', minPoints: 1000, maxPoints: 99999 },
]

export function calculateLevel(points: number): { level: number; title: string; progress: number; nextLevelPoints: number } {
  for (let i = 0; i < LEVEL_THRESHOLDS.length; i++) {
    const tier = LEVEL_THRESHOLDS[i]
    if (points <= tier.maxPoints || i === LEVEL_THRESHOLDS.length - 1) {
      const range = tier.maxPoints - tier.minPoints
      const current = points - tier.minPoints
      const progress = Math.min(100, Math.max(0, Math.round((current / (range || 1)) * 100)))
      return {
        level: tier.level,
        title: tier.name,
        progress,
        nextLevelPoints: tier.maxPoints + 1,
      }
    }
  }
  return { level: 5, title: 'Zero-Waste Champion', progress: 100, nextLevelPoints: 1000 }
}

const DEFAULT_REWARDS_STATE: UserRewardState = {
  points: 45,
  level: 1,
  streakDays: 1,
  lastActiveDate: new Date().toISOString().split('T')[0],
  activities: [
    {
      id: 'init_welcome',
      title: 'Joined EcoSort AI Green Community',
      category: 'general',
      points: 25,
      timestamp: new Date().toISOString(),
      description: 'Welcome bonus for taking steps toward sustainable waste segregation.',
    },
    {
      id: 'init_first_sort',
      title: 'Rinsed & Sorted Clean Cardboard Box',
      category: 'cardboard',
      points: 20,
      timestamp: new Date(Date.now() - 3600000 * 2).toISOString(),
      description: 'Confirmed breaking down shipping box for curbside paper bin.',
    },
  ],
  unlockedBadgeIds: ['first_sort'],
  completedChallengeIds: [],
}

export function getRewardsState(): UserRewardState {
  try {
    const raw = localStorage.getItem(REWARDS_STORAGE_KEY)
    if (!raw) return DEFAULT_REWARDS_STATE
    const parsed = JSON.parse(raw) as UserRewardState
    return parsed
  } catch {
    return DEFAULT_REWARDS_STATE
  }
}

export function saveRewardsState(state: UserRewardState): void {
  try {
    localStorage.setItem(REWARDS_STORAGE_KEY, JSON.stringify(state))
  } catch (err) {
    console.warn('Failed to save rewards state to localStorage', err)
  }
}

export function logConfirmedActivity(activity: Omit<EcoActivity, 'id' | 'timestamp'>): UserRewardState {
  const state = getRewardsState()
  const today = new Date().toISOString().split('T')[0]
  const newActivity: EcoActivity = {
    ...activity,
    id: `act_${Date.now()}`,
    timestamp: new Date().toISOString(),
  }

  // Update streak
  let newStreak = state.streakDays
  if (state.lastActiveDate) {
    const lastDate = new Date(state.lastActiveDate)
    const currentDate = new Date(today)
    const diffDays = Math.round((currentDate.getTime() - lastDate.getTime()) / (1000 * 3600 * 24))
    if (diffDays === 1) {
      newStreak += 1
    } else if (diffDays > 1) {
      newStreak = 1
    }
  } else {
    newStreak = 1
  }

  const newPoints = state.points + activity.points
  const { level } = calculateLevel(newPoints)
  const updatedActivities = [newActivity, ...state.activities]

  // Check badges
  const unlocked = new Set(state.unlockedBadgeIds)
  unlocked.add('first_sort')

  const countCat = (c: WasteCategory) => updatedActivities.filter(a => a.category === c).length
  if (countCat('plastic') >= 5) unlocked.add('plastic_pioneer')
  if (countCat('cardboard') >= 5) unlocked.add('cardboard_crusher')
  if (countCat('metal') >= 5) unlocked.add('metal_maestro')
  if (countCat('glass') >= 5) unlocked.add('glass_guardian')
  if (newStreak >= 3) unlocked.add('streak_3')
  if (newStreak >= 7) unlocked.add('streak_7')

  const nextState: UserRewardState = {
    points: newPoints,
    level,
    streakDays: newStreak,
    lastActiveDate: today,
    activities: updatedActivities,
    unlockedBadgeIds: Array.from(unlocked),
    completedChallengeIds: state.completedChallengeIds,
  }

  saveRewardsState(nextState)
  return nextState
}

export function getDailyChallenges(): DailyChallenge[] {
  const state = getRewardsState()
  const today = new Date().toISOString().split('T')[0]
  const todaysActivities = state.activities.filter(a => a.timestamp.startsWith(today))

  return [
    {
      id: 'daily_rinse',
      title: 'Rinse & Dry Clean 2 Recyclables',
      description: 'Remove food residue from bottles or cans before disposal to prevent contamination.',
      points: 25,
      targetCount: 2,
      currentCount: Math.min(2, todaysActivities.length),
      completed: todaysActivities.length >= 2 || state.completedChallengeIds.includes('daily_rinse'),
      category: 'plastic',
    },
    {
      id: 'daily_cardboard',
      title: 'Flatten 1 Cardboard Box',
      description: 'Break down packaging flat to maximize recycling bin volume.',
      points: 20,
      targetCount: 1,
      currentCount: Math.min(1, todaysActivities.filter(a => a.category === 'cardboard').length),
      completed: todaysActivities.some(a => a.category === 'cardboard') || state.completedChallengeIds.includes('daily_cardboard'),
      category: 'cardboard',
    },
    {
      id: 'daily_quiz',
      title: 'Complete Waste Segregation Quiz',
      description: 'Sharpen your recycling literacy and test your sorting knowledge.',
      points: 30,
      targetCount: 1,
      currentCount: getQuizHistory().length > 0 ? 1 : 0,
      completed: getQuizHistory().length > 0 || state.completedChallengeIds.includes('daily_quiz'),
    },
  ]
}

// ── Quiz Storage ──────────────────────────────────────────────────────────
export function getQuizHistory(): QuizResult[] {
  try {
    const raw = localStorage.getItem(QUIZ_STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function saveQuizResult(result: QuizResult): void {
  try {
    const history = getQuizHistory()
    history.unshift(result)
    localStorage.setItem(QUIZ_STORAGE_KEY, JSON.stringify(history.slice(0, 10)))

    // Award bonus points on first completion
    if (result.percentage >= 80) {
      const state = getRewardsState()
      const unlocked = new Set(state.unlockedBadgeIds)
      if (result.percentage === 100) unlocked.add('quiz_master')

      const updatedState = logConfirmedActivity({
        title: `Eco Quiz Passed (${result.score}/${result.total})`,
        category: 'general',
        points: 40,
        description: `Scored ${result.percentage}% on the Interactive Segregation Quiz.`,
      })
      updatedState.unlockedBadgeIds = Array.from(new Set([...updatedState.unlockedBadgeIds, ...Array.from(unlocked)]))
      saveRewardsState(updatedState)
    }
  } catch (err) {
    console.warn('Failed to save quiz result', err)
  }
}

// ── Environmental Impact Weights & Conversions ─────────────────────────────
// Scientific factors based on US EPA WARM v15 and DEFRA:
// CO2e saved (kg per kg recycled):
//   Metal (Aluminum/Tin): 6.2 kg CO2e/kg
//   Plastic (PET/HDPE): 1.4 kg CO2e/kg
//   Paper/Cardboard: 0.95 kg CO2e/kg
//   Glass: 0.31 kg CO2e/kg
//   Trash diverted: 0.15 kg CO2e/kg (landfill diversion baseline)
export const CO2_FACTORS: Record<WasteCategory, number> = {
  metal: 6.2,
  plastic: 1.4,
  cardboard: 0.95,
  paper: 0.95,
  glass: 0.31,
  trash: 0.15,
}

// Water saved in liters per kg recycled:
export const WATER_FACTORS: Record<WasteCategory, number> = {
  metal: 12.0,
  plastic: 18.0,
  cardboard: 26.0,
  paper: 26.0,
  glass: 2.5,
  trash: 0.5,
}

// Landfill space conserved in cubic meters (m³) per kg:
export const LANDFILL_FACTORS: Record<WasteCategory, number> = {
  plastic: 0.0028,
  cardboard: 0.0024,
  paper: 0.0024,
  metal: 0.0012,
  glass: 0.0008,
  trash: 0.0015,
}

// Trees saved per kg of recycled paper/cardboard (1 ton = ~17 trees => 0.017 trees/kg)
export const TREE_FACTOR_PAPER = 0.017

const DEFAULT_WEIGHT_LOGS: WasteWeightLog[] = [
  {
    id: 'wl_1',
    category: 'cardboard',
    weightGrams: 850,
    loggedAt: new Date(Date.now() - 3600000 * 24 * 3).toISOString(),
    note: 'Flattened grocery & delivery cartons',
  },
  {
    id: 'wl_2',
    category: 'plastic',
    weightGrams: 420,
    loggedAt: new Date(Date.now() - 3600000 * 24 * 2).toISOString(),
    note: 'PET beverage bottles & HDPE milk jug',
  },
  {
    id: 'wl_3',
    category: 'metal',
    weightGrams: 310,
    loggedAt: new Date(Date.now() - 3600000 * 24 * 1).toISOString(),
    note: 'Rinsed aluminum soda cans & soup tin',
  },
  {
    id: 'wl_4',
    category: 'glass',
    weightGrams: 950,
    loggedAt: new Date().toISOString(),
    note: 'Mason jars & olive oil bottle',
  },
]

export function getWeightLogs(): WasteWeightLog[] {
  try {
    const raw = localStorage.getItem(WEIGHT_LOGS_STORAGE_KEY)
    if (!raw) return DEFAULT_WEIGHT_LOGS
    return JSON.parse(raw)
  } catch {
    return DEFAULT_WEIGHT_LOGS
  }
}

export function addWeightLog(log: Omit<WasteWeightLog, 'id' | 'loggedAt'>): WasteWeightLog[] {
  const current = getWeightLogs()
  const newEntry: WasteWeightLog = {
    ...log,
    id: `wl_${Date.now()}`,
    loggedAt: new Date().toISOString(),
  }
  const updated = [newEntry, ...current]
  try {
    localStorage.setItem(WEIGHT_LOGS_STORAGE_KEY, JSON.stringify(updated))
  } catch (err) {
    console.warn('Failed to save weight log', err)
  }

  // Also award eco points for verified weight entry!
  const pointsEarned = Math.max(10, Math.round((log.weightGrams / 100) * 5))
  logConfirmedActivity({
    title: `Logged ${log.weightGrams}g of ${log.category} waste`,
    category: log.category,
    points: pointsEarned,
    description: `Tracked ${log.weightGrams}g in the Environmental Impact Dashboard.`,
  })

  return updated
}

export function deleteWeightLog(id: string): WasteWeightLog[] {
  const current = getWeightLogs()
  const updated = current.filter(l => l.id !== id)
  try {
    localStorage.setItem(WEIGHT_LOGS_STORAGE_KEY, JSON.stringify(updated))
  } catch (err) {
    console.warn('Failed to delete weight log', err)
  }
  return updated
}

export function calculateImpactSummary(logs: WasteWeightLog[]): ImpactSummary {
  let totalWeightGrams = 0
  let co2SavedKg = 0
  let waterSavedLiters = 0
  let paperCardboardKg = 0
  let landfillDivertedM3 = 0

  for (const log of logs) {
    totalWeightGrams += log.weightGrams
    const kg = log.weightGrams / 1000
    co2SavedKg += kg * (CO2_FACTORS[log.category] ?? 0)
    waterSavedLiters += kg * (WATER_FACTORS[log.category] ?? 0)
    landfillDivertedM3 += kg * (LANDFILL_FACTORS[log.category] ?? 0)
    if (log.category === 'paper' || log.category === 'cardboard') {
      paperCardboardKg += kg
    }
  }

  return {
    totalWeightKg: round(totalWeightGrams / 1000, 2),
    co2SavedKg: round(co2SavedKg, 2),
    waterSavedLiters: round(waterSavedLiters, 1),
    treesSaved: round(paperCardboardKg * TREE_FACTOR_PAPER, 3),
    landfillDivertedM3: round(landfillDivertedM3, 4),
  }
}

function round(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals)
  return Math.round(val * factor) / factor
}
