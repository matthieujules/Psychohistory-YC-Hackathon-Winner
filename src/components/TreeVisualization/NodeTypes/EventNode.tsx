'use client';

import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { getSentimentColor } from '@/lib/layout/depth-layout';

interface EventNodeData {
  event: string;
  probability: number;
  sentiment: number;
  depth: number;
}

function EventNode({ data }: { data: EventNodeData }) {
  const { event, probability, sentiment, depth } = data;

  const sentimentColor = getSentimentColor(sentiment);
  const probabilityPercent = (probability * 100).toFixed(1);

  return (
    <div
      className="rounded-lg border-2 bg-white p-3 shadow-lg transition-all hover:shadow-xl"
      style={{
        borderColor: sentimentColor,
        minWidth: 200,
        maxWidth: 250,
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />

      <div className="space-y-2">
        {/* Probability Badge */}
        <div className="flex items-center justify-between">
          <span className="rounded bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-800">
            {probabilityPercent}%
          </span>
          <span className="text-xs text-gray-500">Depth {depth}</span>
        </div>

        {/* Event Text */}
        <p className="text-sm font-medium leading-tight text-gray-900">
          {event.length > 80 ? `${event.substring(0, 80)}...` : event}
        </p>

        {/* Sentiment Bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <span>Sentiment</span>
            <span>{sentiment > 0 ? '+' : ''}{sentiment}</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div
              className="h-2 rounded-full transition-all"
              style={{
                width: `${((sentiment + 100) / 200) * 100}%`,
                backgroundColor: sentimentColor,
              }}
            />
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </div>
  );
}

export default memo(EventNode);
