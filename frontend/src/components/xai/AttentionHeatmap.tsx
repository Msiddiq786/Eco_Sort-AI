/**
 * AttentionHeatmap — displays ViT Attention Rollout heatmap alongside the original image.
 */
import React from 'react';
import { Flame } from 'lucide-react';

interface AttentionHeatmapProps {
  heatmapBase64: string;
  originalImageSrc?: string;
  attentionNote?: string;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({
  heatmapBase64,
  originalImageSrc,
  attentionNote,
}) => {
  return (
    <div className="rounded-xl border border-slate-200/80 bg-white p-4">
      <div className="mb-3 flex items-center gap-2">
        <Flame className="h-4 w-4 text-amber-500" />
        <span className="text-sm font-heading font-bold text-slate-800">Attention Heatmap</span>
        <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-800">
          Attention Rollout
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {originalImageSrc && (
          <div>
            <p className="mb-1.5 text-xs text-slate-500 font-semibold">Original</p>
            <img
              src={originalImageSrc}
              alt="Original waste image"
              className="w-full rounded-xl border border-slate-200 object-contain max-h-56 bg-slate-50"
            />
          </div>
        )}
        <div className={originalImageSrc ? '' : 'sm:col-span-2'}>
          <p className="mb-1.5 text-xs text-slate-500 font-semibold">ViT Attention</p>
          <img
            src={`data:image/png;base64,${heatmapBase64}`}
            alt="ViT Attention Rollout heatmap"
            className="w-full rounded-xl border border-slate-200 object-contain max-h-56 bg-slate-50"
          />
        </div>
      </div>

      {attentionNote && (
        <p className="mt-3 text-xs text-slate-500 leading-relaxed">
          <span className="font-semibold text-slate-700">Method: </span>{attentionNote}
        </p>
      )}
    </div>
  );
};
