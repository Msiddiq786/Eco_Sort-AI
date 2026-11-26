import { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/layout/Layout'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const ClassifyPage = lazy(() => import('@/pages/ClassifyPage'))
const DetectPage = lazy(() => import('@/pages/DetectPage'))
const HistoryPage = lazy(() => import('@/pages/HistoryPage'))
const PerformancePage = lazy(() => import('@/pages/PerformancePage'))
const LearningPage = lazy(() => import('@/pages/LearningPage'))
const RewardsPage = lazy(() => import('@/pages/RewardsPage'))
const DisposalGuidePage = lazy(() => import('@/pages/DisposalGuidePage'))
const ImpactDashboardPage = lazy(() => import('@/pages/ImpactDashboardPage'))
const EnsemblePage = lazy(() => import('@/pages/EnsemblePage'))

function PageFallback() {
  return (
    <div className="space-y-4 py-4">
      <div className="skeleton h-16 w-full rounded-2xl" />
      <div className="skeleton h-64 w-full rounded-2xl" />
    </div>
  )
}

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<PageFallback />}>
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
          <Route path="/ensemble" element={<EnsemblePage />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
