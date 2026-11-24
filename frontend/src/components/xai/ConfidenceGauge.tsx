/**
 * ConfidenceGauge — arc gauge showing calibrated confidence with color coding.
 */
import React from 'react';

interface ConfidenceGaugeProps {
  confidence: number;           // 0.0 – 1.0
  confidenceStatus?: string;    // 'normal' | 'low_confidence' | 'ambiguous' | 'out_of_distribution'
  size?: number;                // px diameter
  showLabel?: boolean;
}

const STATUS_COLORS: Record<string, { stroke: string; badge: string; text: string; label: string }> = {
  normal:             { stroke: '#22C55E', badge: 'bg-green-100 text-green-700',   text: '#16A34A', label: 'Confident' },
  low_confidence:     { stroke: '#F59E0B', badge: 'bg-yellow-100 text-yellow-700', text: '#B45309', label: 'Low Confidence' },
  ambiguous:          { stroke: '#F97316', badge: 'bg-orange-100 text-orange-700', text: '#C2410C', label: 'Ambiguous' },
  out_of_distribution:{ stroke: '#EF4444', badge: 'bg-red-100 text-red-700',       text: '#DC2626', label: 'Out of Distribution' },
};

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  confidenceStatus = 'normal',
  size = 120,
  showLabel = true,
}) => {
  const style = STATUS_COLORS[confidenceStatus] ?? STATUS_COLORS.normal;
  const pct = Math.min(Math.max(confidence, 0), 1);

  // SVG arc gauge (270° sweep from 135° to 45°)
  const radius = 44;
  const cx = 60;
  const cy = 60;
  const sweep = 270;

  const startAngle = 135;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const pointOnCircle = (angle: number) => ({
    x: cx + radius * Math.cos(toRad(angle)),
    y: cy + radius * Math.sin(toRad(angle)),
  });

  const arcPath = (start: number, end: number) => {
    const s = pointOnCircle(start);
    const e = pointOnCircle(end);
    const largeArc = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  };

  const bgEnd = startAngle + sweep;
  const fgEnd = startAngle + pct * sweep;

  return (
    <div className="flex flex-col items-center gap-1" style={{ width: size }}>
      <svg viewBox="0 0 120 120" width={size} height={size}>
        {/* Background track */}
        <path
          d={arcPath(startAngle, bgEnd)}
          fill="none"
          stroke="#E4E4E7"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d={arcPath(startAngle, fgEnd)}
          fill="none"
          stroke={style.stroke}
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Center text */}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="18"
          fontWeight="700"
          fill={style.text}
        >
          {(pct * 100).toFixed(1)}%
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="8"
          fill="#71717A"
        >
          Confidence
        </text>
      </svg>

      {showLabel && (
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${style.badge}`}>
          {style.label}
        </span>
      )}
    </div>
  );
};
