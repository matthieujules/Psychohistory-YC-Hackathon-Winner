/**
 * Core tree generation engine
 */

import { v4 as uuidv4 } from 'uuid';
import { EventNode, SeedInput, TreeState } from '@/types/tree';
import { processNode } from './node-processor';

export class TreeBuilder {
  private state: TreeState;
  private maxConcurrent: number;
  private maxDepth: number;

  constructor(maxDepth = 5, maxConcurrent = 20) {
    this.maxDepth = maxDepth;
    this.maxConcurrent = maxConcurrent;
    this.state = {
      root: null,
      nodeMap: new Map(),
      processingQueue: [],
      completedNodes: new Set(),
      totalNodes: 0,
      maxDepth,
      maxConcurrent,
      isGenerating: false,
    };
  }

  async buildTree(seed: SeedInput): Promise<EventNode> {
    // Create root node
    const root: EventNode = {
      id: uuidv4(),
      event: seed.event,
      probability: 1.0,
      justification: 'Seed event',
      sentiment: 0,
      depth: 0,
      sources: [],
      children: [],
      parentId: null,
      createdAt: new Date(),
      processingStatus: 'pending',
    };

    this.state.root = root;
    this.state.nodeMap.set(root.id, root);
    this.state.processingQueue.push(root.id);
    this.state.totalNodes = 1;
    this.state.isGenerating = true;

    // Expand tree level by level
    await this.expandTree(seed);

    this.state.isGenerating = false;
    return root;
  }

  private async expandTree(seed: SeedInput): Promise<void> {
    let currentDepth = 0;

    while (currentDepth < this.maxDepth && this.state.processingQueue.length > 0) {
      // Get all nodes at current depth
      const nodesAtDepth = this.state.processingQueue.filter(nodeId => {
        const node = this.state.nodeMap.get(nodeId);
        return node && node.depth === currentDepth;
      });

      if (nodesAtDepth.length === 0) {
        currentDepth++;
        continue;
      }

      console.log(
        `Processing depth ${currentDepth} with ${nodesAtDepth.length} nodes`
      );

      // Process nodes in batches of maxConcurrent
      for (let i = 0; i < nodesAtDepth.length; i += this.maxConcurrent) {
        const batch = nodesAtDepth.slice(i, i + this.maxConcurrent);

        await Promise.all(
          batch.map(async nodeId => {
            const node = this.state.nodeMap.get(nodeId);
            if (!node) return;

            try {
              // Process node to generate children
              const children = await processNode(node, seed);

              // Add children to tree
              node.children = children;
              node.processingStatus = 'completed';

              children.forEach(child => {
                this.state.nodeMap.set(child.id, child);
                if (child.depth < this.maxDepth) {
                  this.state.processingQueue.push(child.id);
                }
                this.state.totalNodes++;
              });

              // Remove from queue
              this.state.processingQueue = this.state.processingQueue.filter(
                id => id !== nodeId
              );
              this.state.completedNodes.add(nodeId);

              console.log(
                `Completed node at depth ${node.depth}: ${node.event.substring(0, 50)}...`
              );
            } catch (error) {
              console.error(`Failed to process node ${nodeId}:`, error);
              node.processingStatus = 'failed';
            }
          })
        );
      }

      currentDepth++;
    }

    console.log(
      `Tree generation complete. Total nodes: ${this.state.totalNodes}`
    );
  }

  getState(): TreeState {
    return { ...this.state };
  }

  getNode(id: string): EventNode | undefined {
    return this.state.nodeMap.get(id);
  }
}
