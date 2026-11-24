/**
 * CLIPScoresChart — horizontal bar chart of CLIP similarity scores per waste class.
 */
import React from 'react';
import type { XAICLIPRankedEntry } from '@/types';

interface CLIPScoresChartProps {
  rankedCategories: XAICLIPRankedEntry[];
  topCategory: string;
}

const CATEGORY_COLORS: Record<string, string> = {
  cardboard: '#D97706',
  glass:     '#0EA5E9',
  metal:     '#6B7280',
  paper:     '#8B5CF6',
  plastic:   '#F97316',
  organic:   '#22C55E',
  trash:     '#EF4444',
  unknown:   '#9CA3AF',
};

export const CLIPScoresChart: React.FC<CLIPScoresChartProps> = ({
  rankedCategories,
  topCategory,
}) => {
  const maxScore = Math.max(...rankedCategories.map(e => e.percentage), 0.001);

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-sm font-semibold text-zinc-700">📊 CLIP Similarity Scores</span>
        <span className="rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
          Per Class
        </span>
      </div>

      <div className="space-y-2">
        {rankedCategories.map((entry) => {
          const isTop = entry.category === topCategory;
          const barWidth = `${(entry.percentage / maxScore) * 100}%`;
          const color = CATEGORY_COLORS[entry.category] ?? '#9CA3AF';

          return (
            <div key={entry.category} className="flex items-center gap-2">
              <span
                className="w-20 text-right text-xs font-medium capitalize"
                style={{ color: isTop ? color : '#52525b' }}
              >
                {entry.category}
              </span>
              <div className="relative flex-1 h-5 rounded bg-zinc-100">
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{
                    width: barWidth,
                    backgroundColor: color,
                    opacity: isTop ? 1 : 0.55,
                  }}
                />
                {isTop && (
                  <span className="absolute right-1 top-0.5 text-[10px] font-bold text-white">
                    ★
                  </span>
                )}
              </div>
              <span className="w-12 text-right text-xs text-zinc-500">
                {entry.percentage.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-xs text-zinc-400">
        CLIP computes cosine similarity between the image embedding and text prompts for each waste category.
      </p>
    </div>
  );
};
