import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConfidenceGauge } from '@/components/xai/ConfidenceGauge'

describe('ConfidenceGauge', () => {
  it('renders high confidence status with Confident badge', () => {
    render(<ConfidenceGauge confidence={0.92} confidenceStatus="normal" />)
    expect(screen.getByText('92.0%')).toBeInTheDocument()
    expect(screen.getByText('Confident')).toBeInTheDocument()
  })

  it('renders low confidence badge when confidenceStatus is low_confidence', () => {
    render(<ConfidenceGauge confidence={0.31} confidenceStatus="low_confidence" />)
    expect(screen.getByText('31.0%')).toBeInTheDocument()
    expect(screen.getByText('Low Confidence')).toBeInTheDocument()
  })

  it('renders out-of-distribution badge when confidenceStatus is out_of_distribution', () => {
    render(<ConfidenceGauge confidence={0.22} confidenceStatus="out_of_distribution" />)
    expect(screen.getByText('22.0%')).toBeInTheDocument()
    expect(screen.getByText('Out of Distribution')).toBeInTheDocument()
  })
})
