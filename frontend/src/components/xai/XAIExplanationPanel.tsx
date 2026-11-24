/**
 * XAIExplanationPanel — "Why This Prediction?" section shown on ClassifyPage.
 */
import React, { useState } from 'react';
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
    <div className="rounded-xl border border-zinc-200 bg-zinc-50 shadow-sm overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-zinc-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">🧠</span>
          <span className="text-base font-semibold text-zinc-800">Why This Prediction?</span>
          {explanation.models_agree ? (
            <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
              Models Agree ✓
            </span>
          ) : (
            <span className="rounded-full bg-orange-100 px-2 py-0.5 text-xs font-medium text-orange-700">
              Models Disagree
            </span>
          )}
        </div>
        <span className="text-zinc-400 text-lg">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-4">
          {/* Reasoning text */}
          <div className="rounded-lg bg-white border border-zinc-200 p-4">
            <p
              className="text-sm text-zinc-700 leading-relaxed"
              dangerouslySetInnerHTML={{ __html: explanation.reasoning_text }}
            />
          </div>

          {/* Confidence gauge + ViT/CLIP predictions */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="flex flex-col items-center justify-center rounded-lg border border-zinc-200 bg-white p-4">
              <ConfidenceGauge
                confidence={explanation.confidence}
                confidenceStatus={confidenceStatus}
                size={110}
              />
            </div>

            {/* ViT Explanation */}
            <div className="rounded-lg border border-zinc-200 bg-white p-4">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Vision Transformer (ViT)
              </p>
              <p className="text-sm font-bold capitalize text-zinc-800 mb-1">
                {explanation.vit_explanation.predicted}
              </p>
              <p className="text-xs text-zinc-500 mb-3">
                {(explanation.vit_explanation.confidence * 100).toFixed(1)}% confidence
                · Weight: {(explanation.vit_explanation.weight_in_ensemble * 100).toFixed(0)}%
              </p>
              {explanation.vit_explanation.top3.map((e, i) => (
                <div key={e.category} className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-zinc-400 w-4">#{i + 1}</span>
                  <span className="flex-1 text-xs capitalize text-zinc-700">{e.category}</span>
                  <span className="text-xs font-medium text-zinc-600">
                    {e.percentage.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Ensemble Explanation */}
            <div className="rounded-lg border border-zinc-200 bg-white p-4">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Ensemble Result
              </p>
              <p className="text-xs font-mono text-zinc-500 mb-3 break-all">
                {explanation.ensemble_explanation.formula}
              </p>
              {explanation.ensemble_explanation.top3.map((e, i) => (
                <div key={e.category} className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-zinc-400 w-4">#{i + 1}</span>
                  <div className="flex-1 bg-zinc-100 rounded h-3 relative overflow-hidden">
                    <div
                      className="h-full bg-green-500 rounded"
                      style={{ width: `${e.percentage}%` }}
                    />
                  </div>
                  <span className="text-xs capitalize w-16 text-zinc-700">{e.category}</span>
                  <span className="text-xs font-medium text-zinc-600 w-10 text-right">
                    {e.percentage.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* CLIP Scores Chart */}
          <CLIPScoresChart
            rankedCategories={explanation.clip_explanation.ranked_categories}
            topCategory={explanation.clip_explanation.predicted}
          />

          {/* Attention Heatmap */}
          {explanation.attention_heatmap_base64 ? (
            <AttentionHeatmap
              heatmapBase64={explanation.attention_heatmap_base64}
              originalImageSrc={originalImageSrc}
              attentionNote={explanation.attention_note}
            />
          ) : (
            <div className="rounded-lg border border-dashed border-zinc-300 bg-white p-4 text-center text-sm text-zinc-400">
              Attention heatmap not available for this prediction.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
