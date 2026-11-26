import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AnnotatedImageViewer from '@/components/detect/AnnotatedImageViewer'
import ObjectCard from '@/components/detect/ObjectCard'
import ObjectDetailPanel from '@/components/detect/ObjectDetailPanel'
import type { DetectedObject } from '@/types'

const mockObjectA: DetectedObject = {
  object_id: 1,
  bbox: { x1: 20, y1: 4, x2: 180, y2: 190, nx1: 0.02, ny1: 0.008, nx2: 0.18, ny2: 0.38 },
  detection_confidence: 0.91,
  detection_label: 'Glass item',
  material_category: 'glass',
  vit_prediction: 'glass',
  clip_prediction: 'glass',
  vit_confidence: 0.94,
  clip_confidence: 0.97,
  raw_ensemble_confidence: 0.96,
  final_category: 'glass',
  class_probabilities: { glass: 0.96, plastic: 0.03, trash: 0.01 },
  calibrated_confidence: 0.96,
  confidence_status: 'normal',
  model_agreement: true,
  crop_width: 160,
  crop_height: 186,
  crop_image_base64: 'data:image/jpeg;base64,/9j/4AAQ',
  recycling_status: 'CONDITIONALLY_RECYCLABLE',
  disposal_recommendation: 'Rinse clean and separate metal caps.',
  preparation_steps: ['Rinse clean', 'Remove metal cap'],
  recycling_decision: {
    disposal_status: 'CONDITIONALLY_RECYCLABLE',
    status_label: 'Conditionally Recyclable',
    action: 'Rinse clean and separate caps.',
    preparation_steps: ['Rinse clean', 'Remove metal cap'],
    disposal_bin: 'Blue Bin (Dry Waste / Glass Collection)',
    reason: 'Glass is endlessly recyclable.',
    regional_caveats: 'Follow municipal rules.',
  },
}

const mockObjectB: DetectedObject = {
  ...mockObjectA,
  object_id: 2,
  bbox: { x1: 750, y1: 200, x2: 960, y2: 450, nx1: 0.75, ny1: 0.4, nx2: 0.96, ny2: 0.9 },
  detection_confidence: 0.85,
  detection_label: 'Paper item',
  material_category: 'paper',
  vit_prediction: 'cardboard',
  clip_prediction: 'paper',
  vit_confidence: 0.84,
  clip_confidence: 0.92,
  raw_ensemble_confidence: 0.51,
  final_category: 'paper',
  calibrated_confidence: 0.65,
  confidence_status: 'ambiguous',
  model_agreement: false,
}

describe('Detection UI Components', () => {
  it('renders AnnotatedImageViewer boxes with zoom and selection isolation', () => {
    const onSelect = vi.fn()
    const onFilter = vi.fn()

    const { rerender } = render(
      <AnnotatedImageViewer
        imageBase64="data:image/jpeg;base64,/9j/4AAQ"
        imageWidth={1000}
        imageHeight={500}
        objects={[mockObjectA, mockObjectB]}
        selectedObjectId={null}
        onSelectObject={onSelect}
        materialFilter="all"
        onFilterChange={onFilter}
      />
    )

    // Zoom in button updates zoom display cleanly
    fireEvent.click(screen.getByTitle('Zoom In'))
    expect(screen.getByText('125%')).toBeInTheDocument()

    // Clicking box #1 selects object 1
    fireEvent.click(screen.getByTitle(/Object #1: Glass/i))
    expect(onSelect).toHaveBeenCalledWith(1)

    // When object #1 is selected, Clear Highlight appears and unselected #2 dims
    rerender(
      <AnnotatedImageViewer
        imageBase64="data:image/jpeg;base64,/9j/4AAQ"
        imageWidth={1000}
        imageHeight={500}
        objects={[mockObjectA, mockObjectB]}
        selectedObjectId={1}
        onSelectObject={onSelect}
        materialFilter="all"
        onFilterChange={onFilter}
      />
    )
    expect(screen.getByText('Clear Highlight')).toBeInTheDocument()
    const box2 = screen.getByTitle(/Object #2: Paper/i)
    expect(box2).toHaveStyle({ opacity: '0.35' })
  })

  it('displays independent Detection, ViT, CLIP, and Final Ensemble scores in ObjectCard & ObjectDetailPanel', () => {
    render(
      <div>
        <ObjectCard object={mockObjectA} isSelected={true} onSelect={() => {}} />
        <ObjectDetailPanel object={mockObjectA} onClose={() => {}} />
      </div>
    )

    expect(screen.getByText('Det 91%')).toBeInTheDocument()
    expect(screen.getByText('94%')).toBeInTheDocument()
    expect(screen.getByText('97%')).toBeInTheDocument()
    expect(screen.getByText(/Detection Confidence: 91\.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/Final Ensemble: 96\.0%/i)).toBeInTheDocument()
    expect(screen.getByText('94.0%')).toBeInTheDocument()
    expect(screen.getByText('97.0%')).toBeInTheDocument()
  })
})
