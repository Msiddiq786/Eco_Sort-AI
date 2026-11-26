import { useState } from 'react'
import { Leaf, PlusCircle, Trash2, FileText } from 'lucide-react'
import {
  getWeightLogs,
  deleteWeightLog,
  calculateImpactSummary,
} from '@/services/ecoStorage'
import type { WasteWeightLog } from '@/types'
import { capitalize, CATEGORY_COLORS } from '@/lib'
import PageHeader from '@/components/ui/PageHeader'
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
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="EPA WARM & DEFRA Factors"
        eyebrowIcon={Leaf}
        badge="Local Browser Log"
        icon={Leaf}
        title="Environmental Impact Tracker"
        subtitle="Record diverted material weights locally in your browser to estimate greenhouse gas, water, tree, and landfill savings."
        actions={
          <button
            type="button"
            onClick={() => setModalOpen(true)}
            className="btn-primary text-xs px-4 py-2.5"
          >
            <PlusCircle className="h-4 w-4" />
            <span>Log Recycled Weight</span>
          </button>
        }
      />

      <ImpactSummaryCards summary={summary} />

      <MaterialWeightChart logs={logs} />

      <MethodologyAccordion />

      <div className="card-eco overflow-hidden">
        <div className="section-header">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-ecosort-emerald" />
            <h3 className="section-title">Logged Material Weight History</h3>
          </div>
          <span className="text-xs text-slate-500">{logs.length} logged entries</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs sm:text-sm">
            <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-600">
              <tr>
                <th className="text-left px-5 py-3 font-semibold">Material</th>
                <th className="text-left px-5 py-3 font-semibold">Weight</th>
                <th className="text-left px-5 py-3 font-semibold">Logged Date</th>
                <th className="text-left px-5 py-3 font-semibold">Notes / Description</th>
                <th className="text-right px-5 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {!logs.length ? (
                <tr>
                  <td colSpan={5} className="text-center py-8 text-slate-400">
                    No weight entries recorded yet. Click &quot;Log Recycled Weight&quot; above to start tracking!
                  </td>
                </tr>
              ) : (
                logs.map(log => {
                  const color = CATEGORY_COLORS[log.category] ?? '#64748B'
                  const dateStr = new Date(log.loggedAt).toLocaleDateString()
                  return (
                    <tr key={log.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="px-5 py-3 whitespace-nowrap">
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

                      <td className="px-5 py-3 font-mono font-semibold text-ecosort-emerald whitespace-nowrap">
                        {log.weightGrams >= 1000
                          ? `${(log.weightGrams / 1000).toFixed(2)} kg`
                          : `${log.weightGrams} g`}
                      </td>

                      <td className="px-5 py-3 text-slate-500 whitespace-nowrap">
                        {dateStr}
                      </td>

                      <td className="px-5 py-3 text-slate-600">
                        {log.note || '—'}
                      </td>

                      <td className="px-5 py-3 text-right whitespace-nowrap">
                        <button
                          type="button"
                          onClick={() => handleDelete(log.id)}
                          className="p-1.5 rounded-lg text-rose-600 hover:bg-rose-50 transition-colors"
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
