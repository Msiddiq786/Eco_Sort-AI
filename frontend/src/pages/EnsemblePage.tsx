/**
 * EnsemblePage — Adaptive Ensemble Weight Analysis & Validation Optimization.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Sliders,
  Play,
  Sparkles,
  RefreshCw,
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import PageHeader from '@/components/ui/PageHeader'
import StatusBanner from '@/components/ui/StatusBanner'
import {
  getEnsembleConfig,
  getEnsembleEvaluation,
  optimizeEnsemble,
} from '@/services/api'
import type { EnsembleConfig, EnsembleEvaluation } from '@/types'

export default function EnsemblePage() {
  const [config, setConfig] = useState<EnsembleConfig | null>(null)
  const [evaluation, setEvaluation] = useState<EnsembleEvaluation | null>(null)
  const [loading, setLoading] = useState(true)
  const [optimizing, setOptimizing] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollTimerRef = useRef<number | null>(null)

  const loadData = async () => {
    try {
      const [cfg, evalData] = await Promise.all([
        getEnsembleConfig(),
        getEnsembleEvaluation(),
      ])
      setConfig(cfg)
      setEvaluation(evalData)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Failed to load ensemble configuration.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current)
      }
    }
  }, [])

  const handleOptimize = async () => {
    setError(null)
    setStatusMessage(null)
    setOptimizing(true)
    try {
      const res = await optimizeEnsemble()
      setStatusMessage(res.message)

      const startTime = Date.now()
      const previousOptimizedAt = config?.optimized_at
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current)
      }

      pollTimerRef.current = window.setInterval(async () => {
        try {
          const [cfg, evalData] = await Promise.all([
            getEnsembleConfig(),
            getEnsembleEvaluation(),
          ])
          setConfig(cfg)
          setEvaluation(evalData)

          const finished =
            (cfg.optimized_at && cfg.optimized_at !== previousOptimizedAt) ||
            Date.now() - startTime > 120_000
          if (finished) {
            if (pollTimerRef.current !== null) {
              window.clearInterval(pollTimerRef.current)
              pollTimerRef.current = null
            }
            setOptimizing(false)
            setStatusMessage('Validation grid search complete and active weights updated.')
          }
        } catch {
          // Keep polling until timeout
        }
      }, 3000)
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        'Could not start ensemble optimization.'
      setError(msg)
      setOptimizing(false)
    }
  }

  const chartData = useMemo(() => {
    if (!evaluation?.all_results?.length) return []
    return [...evaluation.all_results]
      .sort((a, b) => a.vit_weight - b.vit_weight)
      .map((row) => ({
        split: `${Math.round(row.vit_weight * 100)}/${Math.round(row.clip_weight * 100)}`,
        accuracyPct: Number((row.accuracy * 100).toFixed(2)),
        macroF1Pct: Number((row.macro_f1 * 100).toFixed(2)),
      }))
  }, [evaluation?.all_results])

  if (loading) {
    return (
      <div className="space-y-6 pb-8">
        <div className="skeleton h-20 w-full rounded-2xl" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((n) => (
            <div key={n} className="skeleton h-32 rounded-2xl" />
          ))}
        </div>
        <div className="skeleton h-64 rounded-2xl" />
      </div>
    )
  }

  const vitPct = Math.round((config?.vit_weight ?? 0.7) * 100)
  const clipPct = Math.round((config?.clip_weight ?? 0.3) * 100)

  const f1Delta =
    evaluation?.best_macro_f1 != null && evaluation?.baseline_macro_f1 != null
      ? evaluation.best_macro_f1 - evaluation.baseline_macro_f1
      : null

  return (
    <div className="space-y-6 pb-10">
      <PageHeader
        eyebrow="Weighted Probability Fusion"
        eyebrowIcon={Sliders}
        icon={Sliders}
        title="Adaptive Ensemble Analysis"
        subtitle="Validation-only grid search over ViT and CLIP fusion weights (zero test-set leakage)."
        actions={
          <>
            <button
              type="button"
              onClick={loadData}
              className="btn-secondary text-xs px-3.5 py-2"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Refresh</span>
            </button>
            <button
              type="button"
              onClick={handleOptimize}
              disabled={optimizing}
              className="btn-primary text-xs px-4 py-2"
            >
              <Play className="h-3.5 w-3.5" />
              <span>{optimizing ? 'Optimizing on Validation Set…' : 'Optimize Weights'}</span>
            </button>
          </>
        }
      />

      {statusMessage && (
        <StatusBanner variant="success" title="Optimization Status">
          {statusMessage}
        </StatusBanner>
      )}

      {error && (
        <StatusBanner variant="error" title="Ensemble API Error">
          {error}
        </StatusBanner>
      )}

      {/* Top Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="stat-card">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Active Fusion Weights
            </span>
            <span
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                config?.source === 'validated'
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-slate-100 text-slate-700'
              }`}
            >
              {config?.source === 'validated' ? 'Validated' : 'Default 70/30'}
            </span>
          </div>
          <p className="text-2xl font-heading font-extrabold text-slate-900">
            {vitPct}% ViT + {clipPct}% CLIP
          </p>
          <div className="mt-2 h-2.5 w-full rounded-full bg-sky-100 overflow-hidden flex">
            <div className="h-full bg-ecosort-emerald" style={{ width: `${vitPct}%` }} />
            <div className="h-full bg-sky-500" style={{ width: `${clipPct}%` }} />
          </div>
          <div className="mt-1.5 flex justify-between text-[11px] text-slate-500 font-medium">
            <span>ViT ({vitPct}%)</span>
            <span>CLIP ({clipPct}%)</span>
          </div>
        </div>

        <div className="stat-card">
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">
            70/30 Baseline (Validation)
          </span>
          <p className="text-2xl font-heading font-extrabold text-slate-900">
            {evaluation?.baseline_macro_f1 != null
              ? `${(evaluation.baseline_macro_f1 * 100).toFixed(2)}% F1`
              : 'Not Run Yet'}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {evaluation?.baseline_accuracy != null
              ? `Validation Accuracy: ${(evaluation.baseline_accuracy * 100).toFixed(2)}%`
              : 'Trigger optimization to compute baseline validation metrics'}
          </p>
        </div>

        <div className="rounded-2xl border border-emerald-200 bg-ecosort-sage p-5 shadow-eco-sm flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-ecosort-forest">
              Best Validated Split
            </span>
            <Sparkles className="h-4 w-4 text-ecosort-emerald" />
          </div>
          {evaluation?.available && evaluation.best_vit_weight != null ? (
            <>
              <p className="text-2xl font-heading font-extrabold text-ecosort-forest">
                {Math.round(evaluation.best_vit_weight * 100)}% ViT +{' '}
                {Math.round((evaluation.best_clip_weight ?? 0) * 100)}% CLIP
              </p>
              <p className="text-xs text-emerald-800 mt-1">
                Macro F1: {((evaluation.best_macro_f1 ?? 0) * 100).toFixed(2)}% · Acc:{' '}
                {((evaluation.best_accuracy ?? 0) * 100).toFixed(2)}%
                {f1Delta !== null && (
                  <span className="ml-1 font-semibold">
                    ({f1Delta >= 0 ? '+' : ''}
                    {(f1Delta * 100).toFixed(2)}% vs 70/30)
                  </span>
                )}
              </p>
              {evaluation.num_samples != null && (
                <p className="text-[11px] text-emerald-700 mt-1">
                  Evaluated on {evaluation.num_samples} held-out validation samples
                </p>
              )}
            </>
          ) : (
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              No grid-search evaluation available yet. Click <strong>Optimize Weights</strong> to sweep ViT/CLIP weights.
            </p>
          )}
        </div>
      </div>

      {/* Validation Curve Visualization */}
      {chartData.length > 0 && (
        <div className="card-eco overflow-hidden">
          <div className="section-header">
            <div>
              <h2 className="section-title">Weight Sweep Tradeoff Curve (ViT / CLIP Ratio)</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Validation Accuracy vs. Macro F1 across fusion ratios from 0/100 (CLIP only) to 100/0 (ViT only)
              </p>
            </div>
          </div>
          <div className="p-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 16, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="split" tick={{ fontSize: 11, fill: '#64748B' }} />
                <YAxis
                  tick={{ fontSize: 11, fill: '#64748B' }}
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  formatter={(val: number, name: string) => [`${val.toFixed(2)}%`, name]}
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #E2E8F0',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  type="monotone"
                  dataKey="macroF1Pct"
                  name="Macro F1 (%)"
                  stroke="#16A34A"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracyPct"
                  name="Accuracy (%)"
                  stroke="#0EA5E9"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Grid Search Results Table */}
      {evaluation?.available && evaluation.all_results && evaluation.all_results.length > 0 && (
        <div className="card-eco overflow-hidden">
          <div className="section-header">
            <h2 className="section-title">
              Validation Weight Sweep Results ({evaluation.all_results.length} configurations)
            </h2>
            <span className="text-xs text-slate-500 font-medium">Selection Criterion: Macro F1</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50/80 border-b border-slate-200 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-5 py-3 text-left font-semibold">ViT Weight</th>
                  <th className="px-5 py-3 text-left font-semibold">CLIP Weight</th>
                  <th className="px-5 py-3 text-right font-semibold">Accuracy</th>
                  <th className="px-5 py-3 text-right font-semibold">Macro F1</th>
                  <th className="px-5 py-3 text-right font-semibold">Tag</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {evaluation.all_results.map((row) => {
                  const isBest =
                    evaluation.best_vit_weight != null &&
                    Math.abs(row.vit_weight - evaluation.best_vit_weight) < 0.001
                  const isBaseline = Math.abs(row.vit_weight - 0.7) < 0.001

                  return (
                    <tr
                      key={`${row.vit_weight}-${row.clip_weight}`}
                      className={isBest ? 'bg-emerald-50/70 font-semibold' : 'hover:bg-slate-50/70'}
                    >
                      <td className="px-5 py-2.5 text-slate-800">
                        {(row.vit_weight * 100).toFixed(0)}%
                      </td>
                      <td className="px-5 py-2.5 text-slate-800">
                        {(row.clip_weight * 100).toFixed(0)}%
                      </td>
                      <td className="px-5 py-2.5 text-right font-mono text-slate-700">
                        {(row.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="px-5 py-2.5 text-right font-mono text-slate-900">
                        {(row.macro_f1 * 100).toFixed(2)}%
                      </td>
                      <td className="px-5 py-2.5 text-right space-x-1.5">
                        {isBest && (
                          <span className="inline-block rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-bold text-emerald-800">
                            Best
                          </span>
                        )}
                        {isBaseline && (
                          <span className="inline-block rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] font-semibold text-slate-700">
                            70/30 Baseline
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
