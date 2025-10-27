'use client';

import { useState } from 'react';
import { SeedInput } from '@/types/tree';
import SeedInputForm from '@/components/SeedInput/SeedForm';
import TreeVisualization from '@/components/TreeVisualization/TreeCanvas';
import { EventNode } from '@/types/tree';

export default function Home() {
  const [tree, setTree] = useState<EventNode | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateTree = async (seed: SeedInput) => {
    setIsGenerating(true);
    setError(null);
    setTree(null);

    try {
      const response = await fetch('/api/generate-tree', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(seed),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to generate tree');
      }

      const data = await response.json();
      setTree(data.tree);
    } catch (err) {
      console.error('Tree generation failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">
            PsychoHistory
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Probabilistic event forecasting powered by historical research
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col lg:flex-row">
        {/* Sidebar */}
        <aside className="w-full border-b border-gray-200 bg-gray-50 lg:w-96 lg:border-b-0 lg:border-r">
          <div className="p-6">
            <SeedInputForm
              onSubmit={handleGenerateTree}
              isLoading={isGenerating}
            />

            {error && (
              <div className="mt-4 rounded-md bg-red-50 p-4">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}
          </div>
        </aside>

        {/* Visualization */}
        <div className="flex-1 bg-gray-100">
          {isGenerating && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <div className="mb-4 inline-block h-12 w-12 animate-spin rounded-full border-4 border-gray-300 border-t-blue-600" />
                <p className="text-gray-600">Generating probability tree...</p>
              </div>
            </div>
          )}

          {!isGenerating && tree && (
            <TreeVisualization tree={tree} />
          )}

          {!isGenerating && !tree && !error && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center text-gray-500">
                <p className="text-lg">Enter a seed event to begin</p>
                <p className="mt-2 text-sm">
                  The system will generate a probability tree of possible outcomes
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
