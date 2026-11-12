import { useState } from 'react'
import { Leaf, PlusCircle, Trash2, FileText } from 'lucide-react'
import {
  getWeightLogs,
  deleteWeightLog,
  calculateImpactSummary,
} from '@/services/ecoStorage'
import type { WasteWeightLog } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib'
import ImpactSummaryCards from '@/components/impact/ImpactSummaryCards'
import MaterialWeightChart from '@/components/impact/MaterialWeightChart'
import MethodologyAccordion from '@/components/impact/MethodologyAccordion'
import WeightLoggerModal from '@/components/impact/WeightLoggerModal'

export default function ImpactDashboardPage() {
  const [logs, setLogs] = useState<WasteWeightLog[]>(getWeightLogs)
  const [modalOpen, setModalOpen] = useState(false)

  const reload = () => {
    setLogs(getWeightLogs())
  }

  const handleDelete = (id: string) => {
    if (!window.confirm('Delete this waste weight entry?')) return
    const updated = deleteWeightLog(id)
    setLogs(updated)
  }

  const summary = calculateImpactSummary(logs)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Leaf className="h-6 w-6 text-primary" />
            Environmental Impact Dashboard
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Track real diverted material quantities and view verifiable greenhouse gas, water, and landfill savings.
          </p>
        </div>

        <button
          onClick={() => setModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-semibold shadow-sm transition-colors self-start sm:self-auto"
        >
          <PlusCircle className="h-4 w-4" /> Log Recycled Weight
        </button>
      </div>

      {/* Summary Cards */}
      <ImpactSummaryCards summary={summary} />

      {/* Material Breakdown & Charts */}
      <MaterialWeightChart logs={logs} />

      {/* Verified Calculation Methodology Accordion */}
      <MethodologyAccordion />

      {/* User Logs Table */}
      <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-emerald-600" />
            <h3 className="font-heading font-bold text-base text-slate-800">
              User Logged Material History
            </h3>
          </div>
          <span className="text-xs text-muted-foreground">
            {logs.length} logged entries
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs sm:text-sm">
            <thead className="bg-slate-50/70 border-b border-border text-slate-600">
              <tr>
                <th className="text-left px-4 py-2.5 font-semibold">Material</th>
                <th className="text-left px-4 py-2.5 font-semibold">Weight</th>
                <th className="text-left px-4 py-2.5 font-semibold">Logged Date</th>
                <th className="text-left px-4 py-2.5 font-semibold">Notes / Description</th>
                <th className="text-right px-4 py-2.5 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {!logs.length ? (
                <tr>
                  <td colSpan={5} className="text-center py-8 text-muted-foreground">
                    No weight entries recorded yet. Click "Log Recycled Weight" above to start tracking!
                  </td>
                </tr>
              ) : (
                logs.map(log => {
                  const color = CATEGORY_COLORS[log.category] ?? '#888'
                  const dateStr = new Date(log.loggedAt).toLocaleDateString()
                  return (
                    <tr key={log.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 rounded-full"
                            style={{ background: color }}
                          />
                          <span className="font-semibold text-slate-800">
                            {capitalize(log.category)}
                          </span>
                        </div>
                      </td>

                      <td className="px-4 py-3 font-mono font-semibold text-emerald-800 whitespace-nowrap">
                        {log.weightGrams >= 1000
                          ? `${(log.weightGrams / 1000).toFixed(2)} kg`
                          : `${log.weightGrams} g`}
                      </td>

                      <td className="px-4 py-3 text-slate-500 whitespace-nowrap">
                        {dateStr}
                      </td>

                      <td className="px-4 py-3 text-slate-600">
                        {log.note || '—'}
                      </td>

                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        <button
                          onClick={() => handleDelete(log.id)}
                          className="p-1 rounded text-rose-600 hover:bg-rose-50 transition-colors"
                          title="Delete Entry"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <WeightLoggerModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={reload}
      />
    </div>
  )
}
