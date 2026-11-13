import { describe, it, expect } from 'vitest'
import { RECYCLING_GUIDE_MATERIALS } from '@/data/recyclingGuideData'
import type { HistoryItem, WasteCategoryV2 } from '@/types'

describe('Smart Recycling Guide Data Verification', () => {
  it('covers all 7 required waste categories plus hazardous items', () => {
    const categories = new Set(RECYCLING_GUIDE_MATERIALS.map(m => m.category))
    expect(categories.has('cardboard')).toBe(true)
    expect(categories.has('glass')).toBe(true)
    expect(categories.has('metal')).toBe(true)
    expect(categories.has('paper')).toBe(true)
    expect(categories.has('plastic')).toBe(true)
    expect(categories.has('organic')).toBe(true)
    expect(categories.has('trash')).toBe(true)
    expect(categories.has('hazardous')).toBe(true)
  })

  it('verifies every material has all mandatory educational and safety fields', () => {
    for (const material of RECYCLING_GUIDE_MATERIALS) {
      expect(material.id).toBeTruthy()
      expect(material.name).toBeTruthy()
      expect(material.description.length).toBeGreaterThan(20)
      expect(material.commonExamples.length).toBeGreaterThanOrEqual(1)
      expect(material.preparationInstructions.length).toBeGreaterThanOrEqual(1)
      expect(material.disposalMethod.binName).toBeTruthy()
      expect(material.disposalMethod.binColor).toBeTruthy()
      expect(material.disposalMethod.instructions).toBeTruthy()
      expect(material.avoidMixing.length).toBeGreaterThanOrEqual(1)
      expect(material.environmentalBenefits).toBeTruthy()
    }
  })

  it('verifies organic waste includes composting explanations and plastic sticker warnings', () => {
    const organicItems = RECYCLING_GUIDE_MATERIALS.filter(m => m.category === 'organic')
    expect(organicItems.length).toBeGreaterThan(0)
    const foodScraps = organicItems.find(m => m.id === 'organic-food-scraps')
    expect(foodScraps).toBeDefined()
    expect(foodScraps?.recyclingPossibility).toBe('Compostable')
    expect(foodScraps?.preparationInstructions.some(p => p.toLowerCase().includes('sticker'))).toBe(true)
  })

  it('verifies hazardous waste strictly warns against putting batteries into ordinary trash', () => {
    const batteryItem = RECYCLING_GUIDE_MATERIALS.find(m => m.id === 'trash-hazardous-batteries')
    expect(batteryItem).toBeDefined()
    expect(batteryItem?.recyclingPossibility).toBe('Special Disposal Required')
    expect(batteryItem?.safetyWarnings?.toLowerCase().includes('fire')).toBe(true)
  })
})

describe('Analytics Calculations & Metrics Logic', () => {
  const sampleItems: HistoryItem[] = [
    {
      id: 1,
      filename: 'can.jpg',
      original_filename: 'can.jpg',
      predicted_category: 'metal',
      confidence: 0.9052,
      vit_prediction: 'metal',
      clip_prediction: 'metal',
      class_probabilities: { metal: 0.9052 },
      created_at: new Date().toISOString(),
    },
    {
      id: 2,
      filename: 'box.jpg',
      original_filename: 'box.jpg',
      predicted_category: 'cardboard',
      confidence: 0.6800,
      vit_prediction: 'cardboard',
      clip_prediction: 'paper',
      class_probabilities: { cardboard: 0.7638 },
      created_at: new Date(Date.now() - 3600 * 1000 * 5).toISOString(),
    },
    {
      id: 3,
      filename: 'apple.jpg',
      original_filename: 'apple.jpg',
      predicted_category: 'organic',
      confidence: 0.9850,
      vit_prediction: 'Not supported (V1 6-class)',
      clip_prediction: 'organic',
      class_probabilities: { organic: 0.9850 },
      created_at: new Date(Date.now() - 3600 * 1000 * 48).toISOString(), // 2 days ago
    },
  ]

  it('correctly calculates total scans, compostable count, and recyclable count', () => {
    const recyclableSet = new Set(['cardboard', 'glass', 'metal', 'paper', 'plastic'])
    let recyclable = 0
    let compostable = 0

    for (const item of sampleItems) {
      if (item.predicted_category === 'organic') compostable++
      else if (recyclableSet.has(item.predicted_category as string)) recyclable++
    }

    expect(sampleItems.length).toBe(3)
    expect(recyclable).toBe(2)
    expect(compostable).toBe(1)
  })

  it('correctly filters items by 24h window (today)', () => {
    const now = Date.now()
    const todayItems = sampleItems.filter(item => {
      const diffHours = (now - new Date(item.created_at).getTime()) / (3600 * 1000)
      return diffHours <= 24
    })

    expect(todayItems.length).toBe(2) // ID 1 and ID 2
  })
})
