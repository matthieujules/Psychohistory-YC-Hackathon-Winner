# PsychoHistory 🔮

Multi-agent future forecasting framework inspired by Isaac Asimov's *Foundation*. PsychoHistory orchestrates dedicated research, reasoning, and presentation agents to map probability-rich futures for complex events and continuously recalibrates itself on real-world case studies.

![PsychoHistory Demo](./demo.png)

## System Capabilities

- Generates branching forecasts up to five layers deep, with calibrated probabilities that sum to one at every decision point.
- Anchors every branch to historical precedent, academic literature, and expert commentary drawn from retrieval agents.
- Scores the qualitative impact of each outcome (sentiment, economic tone, stability risk) to highlight upside, downside, and contrarian paths.
- Streams an interactive tree visualization that allows analysts to drill into any node, inspect rationales, and export structured briefings.

## Operational Flow

1. **Seed Ingestion:** A curator agent accepts the focal event (e.g., “NYC implements rent control”) plus optional context such as timeframe or constraints.
2. **Knowledge Harvesting:** Scout agents query internal corpora and public intelligence feeds, prioritizing primary sources and past analogues.
3. **Hypothesis Formation:** Reasoning agents assemble candidate futures, debate coherence, and assign preliminary likelihoods.
4. **Calibration:** A calibration agent reconciles probabilities across siblings, aligns sentiment scores, and ensures uncertainty is communicated honestly.
5. **Narrative Assembly:** A narrator agent translates the structured tree into analyst-ready explanations with inline citations.
6. **Presentation Layer:** React Flow and D3 render a responsive canvas, while the backend exposes JSON endpoints for downstream tooling.

## Architectural Highlights

- **Frontend:** Next.js App Router, TypeScript, TailwindCSS, Zustand state store, and custom D3 layout logic for depth-aware trees.
- **Backend / AI:** Node edge functions coordinate multi-agent prompts to GPT-4o for synthesis, Exa/Tavily for semantic search, and internal heuristics for probability normalization.
- **State & Contracts:** Strongly typed interfaces (TypeScript + Zod) keep LLM outputs aligned with the visualization and evaluation pipelines.

## Training Regimen

The `training/` workspace fine-tunes a 20B-parameter foundation model so the forecaster improves with each historical replay.

- **Supervised Fine-Tuning (LoRA rank 64):** Runs on NVIDIA A100 40 GB. Thirty curated event chains (Brexit, major regulatory shocks, macro crises) drive each session. ~31.8 M trainable parameters converge in ~335 s with loss 3.136 and perplexity ≈2.8. Checkpoints are stored in `/data/models/sft/final`.
- **Group Relative Policy Optimization (LoRA rank 4):** Executes on NVIDIA A10G 24 GB. For every seed, the system samples 4–8 candidate trees, scores them across calibration, sharpness, diversity, and coherence, then applies group-relative updates to sharpen the distribution.
- **Hot-Swappable Inference:** `modal_inference.py` and `inference.py` load baseline, SFT, or GRPO adapters on demand so evaluation and production can A/B forecasters without rehydrating the base model.

## Data & Evaluation Stack

- **Data Factory:** `data_collection/` and `scripts/` generate JSONL training corpora with explicit depth indices, timestamps, and sentiment annotations. Synthetic datasets ship with the repo; Modal volumes mirror them under `/data`.
- **Telemetry:** `monitor_training.py` and `evaluation/` compute cross-entropy loss, perplexity, match coverage (exact, semantic, LLM-judged), and Brier score per depth level.
- **Drift Sentinels:** Automated checks flag regressions when match coverage drops below 80 %, Brier rises above 0.15, or per-depth perplexity spikes—ensuring the forecaster stays calibrated as new adapters are introduced.

## Interface & Outputs

- **Tree Explorer:** React Flow-based canvas with zoom, pan, depth toggles, and probability overlays; nodes reveal detailed rationales, citations, and sentiment gauges.
- **Analytics Panel:** Aggregated stats for most-probable paths, scenario clusters, and sentiment histograms.
- **Export Layer:** Structured JSON and print-ready narratives support downstream briefing decks or simulation tooling.

## System Blueprint

```
PsychoHistory
├─ Scenario Intake Collective
│  ├─ Curator agent (seed capture + constraints)
│  └─ Cartographer agent (knowledge graph alignment)
├─ Research Engine
│  ├─ Scout agents (historical precedent retrieval)
│  └─ Arbiter agent (source vetting + sentiment read)
├─ Forecast Core
│  ├─ SFT LoRA adapter (A100 fine-tune, rank 64)
│  ├─ GRPO LoRA adapter (A10G refinement, rank 4)
│  └─ Calibration sentinel (loss, perplexity, Brier monitors)
├─ Tree Synthesizer
│  ├─ Branch composer (probability normalization + diversity guardrails)
│  └─ Narrator agent (story weaving with citations)
└─ Analyst Interface
   ├─ Interactive tree explorer (React Flow + D3)
   └─ Briefing generator (JSON + narrative outputs)
```

Built as a one-off hackathon showcase to demonstrate how a tightly coupled multi-agent stack can map future possibility space and self-improve its probabilistic forecasting.
