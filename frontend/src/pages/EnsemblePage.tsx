/**
 * EnsemblePage — Adaptive Ensemble Weight Analysis & Validation Optimization.
 *
 * Displays:
 *  - Active ViT / CLIP weights & configuration source (default vs validated)
 *  - Baseline (70/30) vs best validated configuration comparison
 *  - Validation-only grid search trigger + polling
 *  - Complete weight sweep results table (1.00/0.00 -> 0.00/1.00)
 */
import { useEffect, useRef, useState } from 'react'
import { Sliders, Play, CheckCircle2, AlertCircle, Sparkles, RefreshCw } from 'lucide-react'
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
        Loading ensemble configuration…
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2 text-slate-900">
            <Sliders className="h-6 w-6 text-emerald-600" />
            Adaptive Ensemble Analysis
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Validation-only grid search over ViT and CLIP fusion weights (no test-set leakage).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            onClick={handleOptimize}
            disabled={optimizing}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700 disabled:opacity-60 transition-colors"
          >
            <Play className="h-4 w-4" />
            {optimizing ? 'Optimizing on Validation Set…' : 'Optimize Weights'}
          </button>
        </div>
      </div>

      {statusMessage && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800 flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
          <span>{statusMessage}</span>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Top Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Active Configuration */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
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
          <p className="text-2xl font-extrabold text-slate-900">
            {vitPct}% ViT + {clipPct}% CLIP
          </p>
          <div className="mt-3 h-2.5 w-full rounded-full bg-blue-100 overflow-hidden flex">
            <div className="h-full bg-emerald-600" style={{ width: `${vitPct}%` }} />
            <div className="h-full bg-blue-500" style={{ width: `${clipPct}%` }} />
          </div>
          <div className="mt-2 flex justify-between text-[11px] text-slate-500">
            <span>ViT ({vitPct}%)</span>
            <span>CLIP ({clipPct}%)</span>
          </div>
        </div>

        {/* Baseline 70/30 */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            70/30 Baseline (Validation)
          </span>
          <div className="mt-2 space-y-1">
            <p className="text-2xl font-extrabold text-slate-900">
              {evaluation?.baseline_macro_f1 != null
                ? `${(evaluation.baseline_macro_f1 * 100).toFixed(2)}% F1`
                : 'Not Run Yet'}
            </p>
            <p className="text-xs text-slate-500">
              {evaluation?.baseline_accuracy != null
                ? `Accuracy: ${(evaluation.baseline_accuracy * 100).toFixed(2)}%`
                : 'Trigger optimization to compute baseline metrics'}
            </p>
          </div>
        </div>

        {/* Best Validated Setting */}
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50/40 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-emerald-800">
              Best Validated Split
            </span>
            <Sparkles className="h-4 w-4 text-emerald-600" />
          </div>
          {evaluation?.available && evaluation.best_vit_weight != null ? (
            <>
              <p className="text-2xl font-extrabold text-emerald-950">
                {Math.round(evaluation.best_vit_weight * 100)}% ViT +{' '}
                {Math.round((evaluation.best_clip_weight ?? 0) * 100)}% CLIP
              </p>
              <p className="text-xs text-emerald-800 mt-1">
                Macro F1: {((evaluation.best_macro_f1 ?? 0) * 100).toFixed(2)}% · Accuracy:{' '}
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
            <p className="text-sm text-slate-600 mt-2">
              No grid-search evaluation available yet. Click <strong>Optimize Weights</strong> above.
            </p>
          )}
        </div>
      </div>

      {/* Grid Search Results Table */}
      {evaluation?.available && evaluation.all_results && evaluation.all_results.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900">
              Validation Weight Sweep Results ({evaluation.all_results.length} configurations)
            </h2>
            <span className="text-xs text-slate-500">Selection Criterion: Macro F1</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
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
                      className={isBest ? 'bg-emerald-50/70 font-semibold' : 'hover:bg-slate-50'}
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
                          <span className="inline-block rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-bold text-emerald-800">
                            Best
                          </span>
                        )}
                        {isBaseline && (
                          <span className="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-700">
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
