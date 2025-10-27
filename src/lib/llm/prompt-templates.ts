/**
 * Prompt templates for LLM interactions
 */

export const QUERY_GENERATION_PROMPT = (event: string, context?: string) => `
You are an expert researcher tasked with finding historical precedents and academic research.

Event: ${event}
${context ? `Additional Context: ${context}` : ''}

Generate 3-5 search queries to find:
1. Historical precedents (similar policies, events, or decisions)
2. Academic research on outcomes and effects
3. Expert analysis and predictions
4. Case studies from other countries, cities, or contexts
5. Data-driven reports or statistical analyses

Make queries specific and diverse. Include both positive and negative case studies.

Output ONLY a JSON array of strings:
["query 1", "query 2", "query 3", ...]
`;

export const PROBABILITY_ANALYSIS_PROMPT = (
  parentEvent: string,
  depth: number,
  research: string,
  timeframe?: string
) => `
You are a probabilistic forecasting expert analyzing possible future events.

Parent Event: ${parentEvent}
Current Depth: ${depth}/5
Timeframe: ${timeframe || 'Next significant development'}

Research Findings:
${research}

Based ONLY on the research provided and historical patterns, predict up to 5 possible events that could occur next.

Requirements:
1. Events should be specific, measurable outcomes
2. Probabilities MUST sum to exactly 1.0
3. Each event needs justification citing the research
4. Sentiment: -100 (very negative) to 100 (very positive)
5. Be realistic - include negative outcomes if research suggests them
6. Consider second-order effects

Output ONLY valid JSON (no markdown):
[
  {
    "event": "Specific description of what happens",
    "probability": 0.35,
    "justification": "Based on research showing...",
    "sentiment": -25
  }
]

Probabilities must sum to 1.0. If uncertain, distribute probability more evenly.
`;

export const TIMEFRAME_PROMPT = (event: string, depth: number) => `
Given this event: ${event}
At depth: ${depth}/5

What is a reasonable timeframe for the NEXT significant development?

Output ONLY ONE of:
- "1-2 weeks"
- "1-3 months"
- "3-6 months"
- "6-12 months"
- "1-2 years"

Consider:
- Political/policy events: weeks to months
- Economic shifts: months to years
- Social changes: months to years
- Technological adoption: months to years
`;

export const RESEARCH_SUMMARY_PROMPT = (results: string[]) => `
Synthesize these research findings into a concise summary (max 500 words):

${results.join('\n\n---\n\n')}

Focus on:
1. Key patterns across different cases
2. Success rates and failure modes
3. Timeframes for effects to manifest
4. Common second-order consequences
5. Factors that influenced outcomes

Be objective and balanced. Include both positive and negative findings.
`;
