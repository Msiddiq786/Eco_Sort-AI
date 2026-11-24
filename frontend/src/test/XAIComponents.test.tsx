import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AttentionHeatmap } from '@/components/xai/AttentionHeatmap'
import { CLIPScoresChart } from '@/components/xai/CLIPScoresChart'
import { XAIExplanationPanel } from '@/components/xai/XAIExplanationPanel'
import type { XAIExplanation } from '@/types'

describe('XAI Components', () => {
  it('renders AttentionHeatmap with base64 PNG and method note', () => {
    render(
      <AttentionHeatmap
        heatmapBase64="iVBORw0KGgo="
        originalImageSrc="data:image/jpeg;base64,/9j/4AAQ"
        attentionNote="Bright regions indicate patches that influenced ViT."
      />
    )
    expect(screen.getByText(/Attention Heatmap/i)).toBeInTheDocument()
    expect(screen.getByAltText(/ViT Attention Rollout heatmap/i)).toHaveAttribute(
      'src',
      'data:image/png;base64,iVBORw0KGgo='
    )
    expect(screen.getByText(/Bright regions indicate patches/i)).toBeInTheDocument()
  })

  it('renders CLIPScoresChart with ranked categories', () => {
    render(
      <CLIPScoresChart
        topCategory="plastic"
        rankedCategories={[
          { category: 'plastic', score: 0.82, percentage: 82.0, rank: 1 },
          { category: 'glass', score: 0.12, percentage: 12.0, rank: 2 },
          { category: 'metal', score: 0.06, percentage: 6.0, rank: 3 },
        ]}
      />
    )
    expect(screen.getByText(/CLIP Similarity Scores/i)).toBeInTheDocument()
    expect(screen.getByText('plastic')).toBeInTheDocument()
    expect(screen.getByText('82.0%')).toBeInTheDocument()
    expect(screen.getByText('glass')).toBeInTheDocument()
  })

  it('renders full XAIExplanationPanel with ViT, CLIP, and Ensemble explanations', () => {
    const mockExplanation: XAIExplanation = {
      predicted_category: 'cardboard',
      confidence: 0.91,
      reasoning_text: 'Both the Vision Transformer and CLIP agree that this item is <strong>cardboard</strong>.',
      models_agree: true,
      attention_heatmap_base64: 'iVBORw0KGgo=',
      attention_method: 'attention_rollout',
      attention_note: 'Heatmap generated via Attention Rollout.',
      vit_explanation: {
        predicted: 'cardboard',
        confidence: 0.94,
        weight_in_ensemble: 0.7,
        top3: [
          { category: 'cardboard', score: 0.94, percentage: 94.0 },
          { category: 'paper', score: 0.04, percentage: 4.0 },
          { category: 'trash', score: 0.02, percentage: 2.0 },
        ],
      },
      clip_explanation: {
        predicted: 'cardboard',
        confidence: 0.84,
        weight_in_ensemble: 0.3,
        note: 'CLIP cosine similarity.',
        ranked_categories: [
          { category: 'cardboard', score: 0.84, percentage: 84.0, rank: 1 },
          { category: 'paper', score: 0.16, percentage: 16.0, rank: 2 },
        ],
      },
      ensemble_explanation: {
        formula: 'P_final(c) = 0.7×P_ViT(c) + 0.3×P_CLIP(c)',
        top3: [
          { category: 'cardboard', score: 0.91, percentage: 91.0 },
          { category: 'paper', score: 0.076, percentage: 7.6 },
        ],
      },
    }

    render(<XAIExplanationPanel explanation={mockExplanation} />)
    expect(screen.getByText(/Why This Prediction\?/i)).toBeInTheDocument()
    expect(screen.getByText(/Models Agree/i)).toBeInTheDocument()
    expect(screen.getByText(/P_final\(c\)/i)).toBeInTheDocument()
  })
})
