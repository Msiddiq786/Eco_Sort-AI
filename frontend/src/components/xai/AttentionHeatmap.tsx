/**
 * AttentionHeatmap — displays ViT Attention Rollout heatmap alongside the original image.
 */
import React from 'react';

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
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-sm font-semibold text-zinc-700">🔥 Attention Heatmap</span>
        <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
          Attention Rollout
        </span>
      </div>

      <div className="flex gap-3 overflow-x-auto">
        {originalImageSrc && (
          <div className="flex-1 min-w-0">
            <p className="mb-1 text-xs text-zinc-500 font-medium">Original</p>
            <img
              src={originalImageSrc}
              alt="Original waste image"
              className="w-full rounded object-cover max-h-48"
            />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="mb-1 text-xs text-zinc-500 font-medium">ViT Attention</p>
          <img
            src={`data:image/png;base64,${heatmapBase64}`}
            alt="ViT Attention Rollout heatmap"
            className="w-full rounded object-cover max-h-48"
          />
        </div>
      </div>

      {attentionNote && (
        <p className="mt-3 text-xs text-zinc-500 leading-relaxed">
          <span className="font-medium">Method: </span>{attentionNote}
        </p>
      )}
    </div>
  );
};
