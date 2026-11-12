import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import DashboardPage from '@/pages/DashboardPage'
import ClassifyPage from '@/pages/ClassifyPage'
import DetectPage from '@/pages/DetectPage'
import HistoryPage from '@/pages/HistoryPage'
import PerformancePage from '@/pages/PerformancePage'
import LearningPage from '@/pages/LearningPage'
import RewardsPage from '@/pages/RewardsPage'
import DisposalGuidePage from '@/pages/DisposalGuidePage'
import ImpactDashboardPage from '@/pages/ImpactDashboardPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/classify" element={<ClassifyPage />} />
        <Route path="/detect" element={<DetectPage />} />
        <Route path="/learning" element={<LearningPage />} />
        <Route path="/rewards" element={<RewardsPage />} />
        <Route path="/guide" element={<DisposalGuidePage />} />
        <Route path="/impact" element={<ImpactDashboardPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/performance" element={<PerformancePage />} />
      </Routes>
    </Layout>
  )
}
