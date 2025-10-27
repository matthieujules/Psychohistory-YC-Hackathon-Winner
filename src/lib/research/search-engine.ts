/**
 * Search engine integration (Exa, Perplexity, or fallback)
 */

import { Source, ResearchResult } from '@/types/tree';

export interface SearchConfig {
  provider: 'exa' | 'tavily' | 'mock';
  apiKey?: string;
  maxResults?: number;
  rateLimitPerMinute?: number;
}

// Simple rate limiter
class RateLimiter {
  private queue: number[] = [];
  private limit: number;
  private windowMs: number;

  constructor(limit: number, windowMs: number = 60000) {
    this.limit = limit;
    this.windowMs = windowMs;
  }

  async acquire(): Promise<void> {
    const now = Date.now();

    // Remove old timestamps outside the window
    this.queue = this.queue.filter(timestamp => now - timestamp < this.windowMs);

    // If we've hit the limit, wait
    if (this.queue.length >= this.limit) {
      const oldestRequest = this.queue[0];
      const waitTime = this.windowMs - (now - oldestRequest) + 100; // Add 100ms buffer
      console.log(`Rate limit reached, waiting ${waitTime}ms...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      return this.acquire(); // Recursive retry
    }

    // Record this request
    this.queue.push(now);
  }
}

export class SearchEngine {
  private config: SearchConfig;
  private rateLimiter: RateLimiter;

  constructor(config: SearchConfig) {
    this.config = config;
    // Exa allows 5 queries per second = 300/minute, use conservative 60/minute
    const rateLimitPerMinute = config.rateLimitPerMinute || 60;
    this.rateLimiter = new RateLimiter(rateLimitPerMinute);
  }

  async search(query: string): Promise<Source[]> {
    const { provider, maxResults = 5 } = this.config;

    try {
      // Apply rate limiting for external API calls
      if (provider === 'exa' || provider === 'tavily') {
        await this.rateLimiter.acquire();
      }

      if (provider === 'exa') {
        return await this.searchExa(query, maxResults);
      } else if (provider === 'tavily') {
        return await this.searchTavily(query, maxResults);
      } else {
        return this.mockSearch(query, maxResults);
      }
    } catch (error) {
      console.error(`Search failed for query: ${query}`, error);
      return [];
    }
  }

  private async searchExa(query: string, maxResults: number): Promise<Source[]> {
    // Exa API integration
    const response = await fetch('https://api.exa.ai/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.config.apiKey || '',
      },
      body: JSON.stringify({
        query,
        num_results: maxResults,
        use_autoprompt: true,
        type: 'auto',
      }),
    });

    if (!response.ok) {
      throw new Error(`Exa API error: ${response.statusText}`);
    }

    const data = await response.json();

    return data.results.map((result: any) => ({
      url: result.url,
      title: result.title,
      snippet: result.text || result.snippet || '',
      relevanceScore: result.score,
    }));
  }

  private async searchTavily(query: string, maxResults: number): Promise<Source[]> {
    // Tavily API integration
    const response = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        api_key: this.config.apiKey,
        query,
        search_depth: 'advanced',
        max_results: maxResults,
      }),
    });

    if (!response.ok) {
      throw new Error(`Tavily API error: ${response.statusText}`);
    }

    const data = await response.json();

    return data.results.map((result: any) => ({
      url: result.url,
      title: result.title,
      snippet: result.content,
      relevanceScore: result.score,
    }));
  }

  private mockSearch(query: string, maxResults: number): Source[] {
    // Mock data for testing
    console.log(`Mock search for: ${query}`);

    return Array.from({ length: Math.min(maxResults, 3) }, (_, i) => ({
      url: `https://example.com/research/${i + 1}`,
      title: `Research on ${query} - Study ${i + 1}`,
      snippet: `This study examines ${query} and found significant correlations with historical precedents. Key findings include...`,
      relevanceScore: 0.9 - i * 0.1,
    }));
  }
}

// Default instance
export const defaultSearchEngine = new SearchEngine({
  provider: (process.env.SEARCH_PROVIDER as any) || 'mock',
  apiKey: process.env.EXA_API_KEY || process.env.TAVILY_API_KEY,
  maxResults: 5,
  rateLimitPerMinute: 60, // Exa: 5 QPS = 300/min, conservative 60/min
});

export async function performBatchSearch(queries: string[]): Promise<ResearchResult[]> {
  const results = await Promise.all(
    queries.map(async query => ({
      query,
      sources: await defaultSearchEngine.search(query),
      summary: '',
    }))
  );

  return results.filter(r => r.sources.length > 0);
}
