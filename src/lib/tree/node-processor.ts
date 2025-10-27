/**
 * Process individual nodes: research → probability generation
 */

import { v4 as uuidv4 } from 'uuid';
import { EventNode, SeedInput } from '@/types/tree';
import { generateSearchQueries } from '../llm/query-generator';
import { analyzeProbabilities } from '../llm/probability-analyzer';
import { performBatchSearch } from '../research/search-engine';
import { aggregateResearch, formatResearchForPrompt } from '../research/research-aggregator';

export async function processNode(
  node: EventNode,
  seed: SeedInput
): Promise<EventNode[]> {
  console.log(`Processing node: ${node.event.substring(0, 60)}...`);

  node.processingStatus = 'processing';

  try {
    // Step 1: Generate search queries
    const queries = await generateSearchQueries(
      node.event,
      seed.context
    );

    // Step 2: Perform research
    const searchResults = await performBatchSearch(queries);

    if (searchResults.length === 0) {
      console.warn('No research results found, generating fallback outcomes');
      return generateFallbackChildren(node);
    }

    // Step 3: Aggregate research
    const research = await aggregateResearch(searchResults);
    const researchText = formatResearchForPrompt(research);

    // Step 4: Generate probabilities
    const probabilities = await analyzeProbabilities(
      node.event,
      node.depth,
      researchText,
      seed.timeframe
    );

    // Step 5: Create child nodes
    const children: EventNode[] = probabilities.map(prob => ({
      id: uuidv4(),
      event: prob.event,
      probability: prob.probability,
      justification: prob.justification,
      sentiment: prob.sentiment,
      depth: node.depth + 1,
      sources: research.results.flatMap(r => r.sources).slice(0, 5),
      children: [],
      parentId: node.id,
      createdAt: new Date(),
      processingStatus: 'pending',
    }));

    console.log(`Generated ${children.length} children for node at depth ${node.depth}`);

    return children;
  } catch (error) {
    console.error('Node processing failed:', error);
    return generateFallbackChildren(node);
  }
}

function generateFallbackChildren(node: EventNode): EventNode[] {
  // Fallback when research/LLM fails
  return [
    {
      id: uuidv4(),
      event: `Status quo continues from: ${node.event.substring(0, 50)}`,
      probability: 0.5,
      justification: 'No significant change detected',
      sentiment: 0,
      depth: node.depth + 1,
      sources: [],
      children: [],
      parentId: node.id,
      createdAt: new Date(),
      processingStatus: 'pending',
    },
    {
      id: uuidv4(),
      event: `Unexpected development from: ${node.event.substring(0, 50)}`,
      probability: 0.5,
      justification: 'Insufficient data for detailed prediction',
      sentiment: -10,
      depth: node.depth + 1,
      sources: [],
      children: [],
      parentId: node.id,
      createdAt: new Date(),
      processingStatus: 'pending',
    },
  ];
}
