/**
 * XAIExplanationPanel — "Why This Prediction?" section shown on ClassifyPage & ObjectDetailPanel.
 */
import React, { useState } from 'react';
import { BrainCircuit, ChevronDown, ChevronUp, CheckCircle2, AlertTriangle } from 'lucide-react';
import type { XAIExplanation } from '@/types';
import { AttentionHeatmap } from './AttentionHeatmap';
import { CLIPScoresChart } from './CLIPScoresChart';
import { ConfidenceGauge } from './ConfidenceGauge';

interface XAIExplanationPanelProps {
  explanation: XAIExplanation;
  originalImageSrc?: string;
  confidenceStatus?: string;
}

export const XAIExplanationPanel: React.FC<XAIExplanationPanelProps> = ({
  explanation,
  originalImageSrc,
  confidenceStatus = 'normal',
}) => {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="card-eco overflow-hidden animate-fade-in">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-5 py-4 bg-ecosort-sage/70 hover:bg-ecosort-mint transition-colors border-b border-slate-100"
      >
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-ecosort-forest text-ecosort-lime">
            <BrainCircuit className="h-4 w-4" />
          </span>
          <span className="text-base font-heading font-bold text-slate-900">
            Why This Prediction?
          </span>
          {explanation.models_agree ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-800">
              <CheckCircle2 className="h-3 w-3" />
              Models Agree
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-800">
              <AlertTriangle className="h-3 w-3" />
              Models Disagree
            </span>
          )}
        </div>
        <span className="text-slate-500">
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </span>
      </button>

      {expanded && (
        <div className="p-5 space-y-4 bg-slate-50/40">
          <div className="rounded-xl bg-white border border-slate-200/80 p-4">
            <p
              className="text-sm text-slate-700 leading-relaxed"
              dangerouslySetInnerHTML={{ __html: explanation.reasoning_text }}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="flex flex-col items-center justify-center rounded-xl border border-slate-200/80 bg-white p-4">
              <ConfidenceGauge
                confidence={explanation.confidence}
                confidenceStatus={confidenceStatus}
                size={110}
              />
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-white p-4">
              <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">
                Vision Transformer (ViT)
              </p>
              <p className="text-sm font-heading font-extrabold capitalize text-slate-900 mb-1">
                {explanation.vit_explanation.predicted}
              </p>
              <p className="text-xs text-slate-500 mb-3">
                {(explanation.vit_explanation.confidence * 100).toFixed(1)}% confidence
                · Weight: {(explanation.vit_explanation.weight_in_ensemble * 100).toFixed(0)}%
              </p>
              {explanation.vit_explanation.top3.map((e, i) => (
                <div key={e.category} className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-slate-400 w-4 font-mono">#{i + 1}</span>
                  <span className="flex-1 text-xs capitalize text-slate-700">{e.category}</span>
                  <span className="text-xs font-mono font-semibold text-slate-700">
                    {e.percentage.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            <div className="rounded-xl border border-slate-200/80 bg-white p-4">
              <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">
                Ensemble Result
              </p>
              <p className="text-xs font-mono text-slate-500 mb-3 break-all">
                {explanation.ensemble_explanation.formula}
              </p>
              {explanation.ensemble_explanation.top3.map((e, i) => (
                <div key={e.category} className="flex items-center gap-2 mb-1.5">
                  <span className="text-xs text-slate-400 w-4 font-mono">#{i + 1}</span>
                  <div className="flex-1 bg-slate-100 rounded-full h-2.5 relative overflow-hidden">
                    <div
                      className="h-full bg-ecosort-emerald rounded-full"
                      style={{ width: `${e.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs capitalize w-16 text-slate-700 truncate">{e.category}</span>
                  <span className="text-xs font-mono font-semibold text-slate-700 w-10 text-right">
                    {e.percentage.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <CLIPScoresChart
            rankedCategories={explanation.clip_explanation.ranked_categories}
            topCategory={explanation.clip_explanation.predicted}
          />

          {explanation.attention_heatmap_base64 ? (
            <AttentionHeatmap
              heatmapBase64={explanation.attention_heatmap_base64}
              originalImageSrc={originalImageSrc}
              attentionNote={explanation.attention_note}
            />
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-center text-sm text-slate-400">
              Attention heatmap not available for this prediction.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
