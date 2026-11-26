import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import DropZone from '@/components/classify/DropZone'
import FeedbackCorrection from '@/components/feedback/FeedbackCorrection'
import ObjectDetailPanel from '@/components/detect/ObjectDetailPanel'
import ResultCard from '@/components/classify/ResultCard'
import HistoryPage from '@/pages/HistoryPage'
import * as api from '@/services/api'
import type { DetectedObject, PredictResponse } from '@/types'

vi.mock('@/services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/services/api')>()
  return {
    ...actual,
    submitFeedback: vi.fn(),
    explainImage: vi.fn(),
    getRecommendation: vi.fn(),
    getHistory: vi.fn(),
    getDetectionHistory: vi.fn(),
    deleteHistoryItem: vi.fn(),
    deleteDetectionHistoryItem: vi.fn(),
  }
})

const sampleObject: DetectedObject = {
  object_id: 1,
  bbox: { x1: 10, y1: 15, x2: 110, y2: 155, nx1: 0.05, ny1: 0.05, nx2: 0.55, ny2: 0.75 },
  detection_confidence: 0.91,
  detection_label: 'Plastic item',
  material_category: 'plastic',
  vit_prediction: 'plastic',
  clip_prediction: 'plastic',
  vit_confidence: 0.9,
  clip_confidence: 0.93,
  raw_ensemble_confidence: 0.92,
  final_category: 'plastic',
  class_probabilities: { plastic: 0.92, glass: 0.05, trash: 0.03 },
  calibrated_confidence: 0.92,
  confidence_status: 'normal',
  model_agreement: true,
  crop_width: 100,
  crop_height: 140,
  crop_image_base64: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/',
  recycling_status: 'RECYCLABLE',
  disposal_recommendation: 'Empty liquid and place in Blue Bin.',
  preparation_steps: ['Rinse clean'],
  recycling_decision: {
    disposal_status: 'RECYCLABLE',
    status_label: 'Recyclable',
    action: 'Empty liquid and place in Blue Bin.',
    preparation_steps: ['Rinse clean'],
    disposal_bin: 'Blue Bin (Dry Waste)',
    reason: 'PET plastic bottles are recyclable.',
    regional_caveats: 'Follow municipal rules.',
  },
  resin_info: {
    resin_code: '#1 PET',
    resin_number: 1,
    resin_name: 'Polyethylene Terephthalate (PET / PETE)',
    recyclability_tier: 'Widely Recyclable',
    confidence: 0.86,
    hazard_flags: [],
    method: 'clip_roi',
    guidance: 'Empty liquid, flatten bottle, and keep cap on if accepted locally.',
  },
}

const samplePrediction: PredictResponse = {
  id: 42,
  filename: 'bottle.jpg',
  predicted_category: 'plastic',
  confidence: 0.89,
  vit_prediction: 'plastic',
  clip_prediction: 'plastic',
  class_probabilities: {
    cardboard: 0.02,
    glass: 0.03,
    metal: 0.02,
    paper: 0.02,
    plastic: 0.89,
    trash: 0.02,
  },
  model_agreement: true,
  recycling_recommendation: 'Rinse and place in plastic recycling.',
  resin_info: {
    resin_code: '#5 PP',
    resin_number: 5,
    resin_name: 'Polypropylene (PP)',
    recyclability_tier: 'Conditionally Recyclable',
    confidence: 0.78,
    hazard_flags: [],
    method: 'clip_roi',
    guidance: 'Rinse food residue completely before recycling.',
  },
  model_version: '1.0.0',
  created_at: new Date().toISOString(),
}

describe('EcoSort AI V3 Sprint Features UI', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getRecommendation).mockResolvedValue({
      category: 'plastic',
      bin_color: '#2563EB',
      bin_label: 'Blue Bin',
      stream: 'Dry Recyclable',
      disposal_status: 'RECYCLABLE',
      status_label: 'Recyclable',
      action: 'Rinse and recycle.',
      preparation_steps: ['Rinse clean'],
      eco_tip: 'Flatten bottles.',
      regional_caveats: 'Check local guidelines.',
    })
  })

  it('supports switching DropZone into Live Camera mode safely', async () => {
    const onFile = vi.fn()
    render(<DropZone onFile={onFile} loading={false} />)

    expect(screen.getByText('Upload Image')).toBeInTheDocument()
    const cameraBtn = screen.getByText('Live Camera')
    fireEvent.click(cameraBtn)

    expect(await screen.findByText('Capture & Analyze Frame')).toBeInTheDocument()
  })

  it('submits user label correction through FeedbackCorrection component', async () => {
    vi.mocked(api.submitFeedback).mockResolvedValue({
      id: 7,
      predicted_category: 'cardboard',
      corrected_category: 'paper',
      source: 'single',
      created_at: new Date().toISOString(),
      message: "Feedback saved: 'cardboard' -> 'paper'.",
    })

    render(
      <FeedbackCorrection
        predictedCategory="cardboard"
        predictionId={12}
        source="single"
      />
    )

    fireEvent.click(screen.getByText('Correct Label'))
    const select = screen.getByLabelText('Actual Material Class')
    fireEvent.change(select, { target: { value: 'paper' } })
    fireEvent.click(screen.getByText('Save Correction'))

    await waitFor(() => {
      expect(api.submitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          predicted_category: 'cardboard',
          corrected_category: 'paper',
          prediction_id: 12,
          source: 'single',
        })
      )
      expect(screen.getByText(/Feedback saved: 'cardboard' -> 'paper'/)).toBeInTheDocument()
    })
  })

  it('displays Resin Code badge and triggers per-object XAI inside ObjectDetailPanel', async () => {
    vi.mocked(api.explainImage).mockResolvedValue({
      predicted_category: 'plastic',
      confidence: 0.92,
      reasoning_text: 'Highlighted plastic neck and translucent bottle contour.',
      models_agree: true,
      attention_heatmap_base64: 'data:image/png;base64,iVBORw0KGgo=',
      attention_method: 'attention_rollout',
      attention_note: 'ViT rollout',
      vit_explanation: {
        predicted: 'plastic',
        confidence: 0.9,
        weight_in_ensemble: 0.7,
        top3: [{ category: 'plastic', score: 0.9, percentage: 90 }],
      },
      clip_explanation: {
        predicted: 'plastic',
        confidence: 0.93,
        weight_in_ensemble: 0.3,
        note: 'CLIP similarity',
        ranked_categories: [{ category: 'plastic', score: 0.93, percentage: 93, rank: 1 }],
      },
      ensemble_explanation: {
        formula: '0.48 ViT + 0.52 CLIP',
        top3: [{ category: 'plastic', score: 0.92, percentage: 92 }],
      },
    })

    render(<ObjectDetailPanel object={sampleObject} scanId={9} onClose={() => {}} />)

    expect(screen.getByText('#1 PET')).toBeInTheDocument()
    expect(screen.getByText(/Polyethylene Terephthalate/)).toBeInTheDocument()

    fireEvent.click(screen.getByText('Explain Object #1 (XAI)'))

    await waitFor(() => {
      expect(api.explainImage).toHaveBeenCalledTimes(1)
      expect(screen.getByText('Why This Prediction?')).toBeInTheDocument()
    })
  })

  it('renders Resin badge and FeedbackCorrection inside ResultCard', async () => {
    render(<ResultCard result={samplePrediction} />)
    expect(screen.getByText(/#5 PP/)).toBeInTheDocument()
    expect(screen.getByText('Correct Label')).toBeInTheDocument()
    expect(await screen.findByText(/Rinse and recycle\./)).toBeInTheDocument()
  })

  it('shows persisted Multi-Object Audits tab inside HistoryPage', async () => {
    vi.mocked(api.getHistory).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 15,
    })
    vi.mocked(api.getDetectionHistory).mockResolvedValue({
      items: [
        {
          id: 3,
          filename: 'audit_uuid.jpg',
          original_filename: 'kitchen_mix.jpg',
          image_path: 'uploads/audit_uuid.jpg',
          total_objects: 2,
          category_counts: { glass: 1, metal: 1 },
          status_counts: { recyclable: 2 },
          objects: [
            {
              object_id: 1,
              material_category: 'glass',
              calibrated_confidence: 0.94,
              vit_prediction: 'glass',
              clip_prediction: 'glass',
              recycling_status: 'CONDITIONALLY_RECYCLABLE',
              model_agreement: true,
            },
          ],
          inference_time_ms: 420.5,
          detection_model_status: 'hybrid_yolov8',
          created_at: new Date().toISOString(),
        },
      ],
      total: 1,
      page: 1,
      page_size: 15,
    })

    render(<HistoryPage />)

    const auditTab = await screen.findByText(/Multi-Object Audits \(1\)/)
    fireEvent.click(auditTab)

    expect(await screen.findByText('Audit #3')).toBeInTheDocument()
    expect(screen.getByText('kitchen_mix.jpg')).toBeInTheDocument()
  })
})
