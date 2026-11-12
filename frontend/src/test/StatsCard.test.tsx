import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatsCard from '@/components/dashboard/StatsCard'
import { Image as ImageIcon } from 'lucide-react'

describe('StatsCard', () => {
  it('renders title and value', () => {
    render(
      <StatsCard
        title="Total Classified"
        value="42"
        icon={<ImageIcon data-testid="icon" />}
        description="images processed"
      />
    )
    expect(screen.getByText('Total Classified')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('images processed')).toBeInTheDocument()
  })
})
