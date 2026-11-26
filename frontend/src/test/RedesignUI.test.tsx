import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ScanLine } from 'lucide-react'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
import EmptyState from '@/components/ui/EmptyState'
import CategoryBadge from '@/components/ui/CategoryBadge'
import LearningPage from '@/pages/LearningPage'
import { resolveApiBaseUrl } from '@/services/api'

describe('EcoSort UI Design System Components', () => {
  it('renders PageHeader with eyebrow, badge, title, and subtitle', () => {
    render(
      <PageHeader
        eyebrow="Live Telemetry"
        badge="V3"
        icon={ScanLine}
        title="Waste Analytics"
        subtitle="Track dual-model ensemble outputs."
      />
    )
    expect(screen.getByText('Live Telemetry')).toBeInTheDocument()
    expect(screen.getByText('V3')).toBeInTheDocument()
    expect(screen.getByText('Waste Analytics')).toBeInTheDocument()
    expect(screen.getByText('Track dual-model ensemble outputs.')).toBeInTheDocument()
  })

  it('renders StatusBanner variants with accessible role', () => {
    render(
      <StatusBanner variant="error" title="Connection Error">
        Backend unavailable
      </StatusBanner>
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('Connection Error')).toBeInTheDocument()
    expect(screen.getByText('Backend unavailable')).toBeInTheDocument()
  })

  it('renders EmptyState and CategoryBadge', () => {
    render(
      <div>
        <EmptyState
          icon={ScanLine}
          title="No Scans Yet"
          description="Upload an image to begin."
        />
        <CategoryBadge category="organic" />
      </div>
    )
    expect(screen.getByText('No Scans Yet')).toBeInTheDocument()
    expect(screen.getByText('Organic')).toBeInTheDocument()
  })

  it('renders all 7 waste categories including Organic in LearningPage', () => {
    render(
      <MemoryRouter>
        <LearningPage />
      </MemoryRouter>
    )
    expect(screen.getByText('7 Waste Categories')).toBeInTheDocument()
    expect(screen.getByText('Organic')).toBeInTheDocument()
    expect(screen.getByText('Compostable Organics')).toBeInTheDocument()
    expect(screen.getByText('Cardboard')).toBeInTheDocument()
  })

  it('resolves production API URL cleanly with or without trailing slash', () => {
    expect(resolveApiBaseUrl(undefined)).toBe('/api')
    expect(resolveApiBaseUrl('https://ecosort-api.onrender.com/')).toBe('https://ecosort-api.onrender.com/api')
    expect(resolveApiBaseUrl('https://ecosort-api.up.railway.app/api')).toBe('https://ecosort-api.up.railway.app/api')
  })
})
