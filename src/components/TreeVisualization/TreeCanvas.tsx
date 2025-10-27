'use client';

import { useCallback, useMemo, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import * as d3 from 'd3';

import { EventNode } from '@/types/tree';
import { calculateTreeLayout } from '@/lib/layout/depth-layout';
import EventNodeComponent from './NodeTypes/EventNode';
import SeedNodeComponent from './NodeTypes/SeedNode';
import ProbabilityEdge from './EdgeTypes/ProbabilityEdge';
import NodeDetailsPanel from './NodeDetailsPanel';

interface Props {
  tree: EventNode;
}

const nodeTypes = {
  event: EventNodeComponent,
  seed: SeedNodeComponent,
};

const edgeTypes = {
  probability: ProbabilityEdge,
};

export default function TreeVisualization({ tree }: Props) {
  const [selectedNode, setSelectedNode] = useState<EventNode | null>(null);
  const [orientation, setOrientation] = useState<'vertical' | 'horizontal'>('vertical');

  // Calculate layout (recalculates whenever tree changes)
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () =>
      calculateTreeLayout(tree, {
        depthSpacing: 300,
        childSpacing: 200,
        orientation,
      }),
    [tree, orientation]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes/edges when tree changes
  useEffect(() => {
    const { nodes: newNodes, edges: newEdges } = calculateTreeLayout(tree, {
      depthSpacing: 300,
      childSpacing: 200,
      orientation,
    });

    setNodes(newNodes);
    setEdges(newEdges);
  }, [tree, orientation, setNodes, setEdges]);

  const onNodeClick = useCallback(
    (_: any, node: any) => {
      const eventNode = findNodeById(tree, node.id);
      setSelectedNode(eventNode);
    },
    [tree]
  );

  const toggleOrientation = () => {
    setOrientation(prev => (prev === 'vertical' ? 'horizontal' : 'vertical'));
  };

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        minZoom={0.1}
        maxZoom={2}
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={node => {
            const sentiment = node.data?.sentiment || 0;
            // Use D3 color scale for minimap
            return d3.interpolateRdYlGn((sentiment + 100) / 200);
          }}
        />

        <Panel position="top-right" className="space-x-2">
          <button
            onClick={toggleOrientation}
            className="rounded-md bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-md hover:bg-gray-50"
          >
            {orientation === 'vertical' ? 'Vertical' : 'Horizontal'}
          </button>
        </Panel>
      </ReactFlow>

      {selectedNode && (
        <NodeDetailsPanel
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}

function findNodeById(root: EventNode, id: string): EventNode | null {
  if (root.id === id) return root;

  for (const child of root.children) {
    const found = findNodeById(child, id);
    if (found) return found;
  }

  return null;
}
