'use client';

import { useCallback, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { EventNode } from '@/types/tree';
import { calculateTreeLayout } from '@/lib/layout/depth-layout';
import EventNodeComponent from './NodeTypes/EventNode';
import SeedNodeComponent from './NodeTypes/SeedNode';
import NodeDetailsPanel from './NodeDetailsPanel';

interface Props {
  tree: EventNode;
}

const nodeTypes = {
  event: EventNodeComponent,
  seed: SeedNodeComponent,
};

export default function TreeVisualization({ tree }: Props) {
  const [selectedNode, setSelectedNode] = useState<EventNode | null>(null);
  const [orientation, setOrientation] = useState<'vertical' | 'horizontal'>('vertical');

  // Calculate layout
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
        fitView
        minZoom={0.1}
        maxZoom={2}
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={node => {
            const sentiment = node.data?.sentiment || 0;
            if (sentiment > 50) return '#22c55e';
            if (sentiment > 0) return '#84cc16';
            if (sentiment > -50) return '#f97316';
            return '#ef4444';
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
