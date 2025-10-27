'use client';

import { memo } from 'react';
import { EdgeProps, getBezierPath } from 'reactflow';
import { getEdgeWidth, getEdgeColor, getSentimentColor } from '@/lib/d3/color-scales';

interface ProbabilityEdgeData {
  probability: number;
  sentiment?: number;
}

function ProbabilityEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps<ProbabilityEdgeData>) {
  const probability = data?.probability || 0.5;
  const sentiment = data?.sentiment || 0;

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const strokeWidth = getEdgeWidth(probability);
  const color = getEdgeColor(probability, sentiment);
  const sentimentColor = getSentimentColor(sentiment);

  // High probability edges get a glow effect
  const shouldGlow = probability > 0.5;
  const glowIntensity = probability > 0.7 ? 'strong' : probability > 0.5 ? 'medium' : 'none';

  return (
    <g className="react-flow__edge">
      {/* Outer glow for high-probability edges */}
      {shouldGlow && (
        <>
          <path
            d={edgePath}
            fill="none"
            stroke={sentimentColor}
            strokeWidth={strokeWidth + 8}
            strokeOpacity={probability * 0.15}
            className="transition-all duration-300"
          />
          <path
            d={edgePath}
            fill="none"
            stroke={sentimentColor}
            strokeWidth={strokeWidth + 4}
            strokeOpacity={probability * 0.25}
            className="transition-all duration-300"
          />
        </>
      )}

      {/* Main edge path */}
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        className="transition-all duration-300"
        markerEnd={markerEnd}
        style={{
          filter: glowIntensity === 'strong'
            ? `drop-shadow(0 0 4px ${sentimentColor})`
            : undefined,
        }}
      />

      {/* Animated particles for very high probability paths */}
      {probability > 0.8 && (
        <>
          <circle r="2" fill={sentimentColor} opacity="0.8">
            <animateMotion
              dur="2s"
              repeatCount="indefinite"
              path={edgePath}
            />
          </circle>
          <circle r="2" fill={sentimentColor} opacity="0.6">
            <animateMotion
              dur="2s"
              begin="0.5s"
              repeatCount="indefinite"
              path={edgePath}
            />
          </circle>
        </>
      )}
    </g>
  );
}

export default memo(ProbabilityEdge);
