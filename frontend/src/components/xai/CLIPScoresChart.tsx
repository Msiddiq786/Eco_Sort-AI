/**
 * CLIPScoresChart — horizontal bar chart of CLIP similarity scores per waste class.
 */
import React from 'react';
import { BarChart3 } from 'lucide-react';
import type { XAICLIPRankedEntry } from '@/types';
import { CATEGORY_COLORS as ECO_COLORS } from '@/lib/utils';

interface CLIPScoresChartProps {
  rankedCategories: XAICLIPRankedEntry[];
  topCategory: string;
}

export const CLIPScoresChart: React.FC<CLIPScoresChartProps> = ({
  rankedCategories,
  topCategory,
}) => {
  const maxScore = Math.max(...rankedCategories.map(e => e.percentage), 0.001);

  return (
    <div className="rounded-xl border border-slate-200/80 bg-white p-4">
      <div className="mb-3 flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-sky-600" />
        <span className="text-sm font-heading font-bold text-slate-800">CLIP Similarity Scores</span>
        <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[11px] font-semibold text-sky-800">
          Per Class
        </span>
      </div>

      <div className="space-y-2">
        {rankedCategories.map((entry) => {
          const isTop = entry.category === topCategory;
          const barWidth = `${(entry.percentage / maxScore) * 100}%`;
          const color = ECO_COLORS[entry.category.toLowerCase()] ?? '#64748B';

          return (
            <div key={entry.category} className="flex items-center gap-2.5">
              <span
                className="w-20 text-right text-xs font-semibold capitalize"
                style={{ color: isTop ? color : '#475569' }}
              >
                {entry.category}
              </span>
              <div className="relative flex-1 h-4 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: barWidth,
                    backgroundColor: color,
                    opacity: isTop ? 1 : 0.6,
                  }}
                />
              </div>
              <span className="w-12 text-right text-xs font-mono font-semibold text-slate-600">
                {entry.percentage.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      <p className="mt-3 text-xs text-slate-400">
        CLIP computes cosine similarity between the image embedding and class text prompts for each waste category.
      </p>
    </div>
  );
};
