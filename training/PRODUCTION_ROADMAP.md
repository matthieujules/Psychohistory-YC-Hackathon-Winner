# Production Training Roadmap

## What Actually Happened in This Run

### Training Data (Synthetic)
- **Source:** 10 templated historical scenarios
- **Format:** Manually created in `scripts/generate_synthetic_data.py`
- **Examples:** Tech layoffs, interest rate changes, UBI pilots, social media bans, etc.
- **Quality:** Simplified, predictable patterns
- **Purpose:** Test the training pipeline end-to-end

### Training Process
```
Input (Prompt):
  "Given this historical event:
   Event: Major tech company announces layoffs of 10,000 employees
   Date: 2023-01-15
   Context: Tech downturn, rising interest rates, over-hiring during pandemic

   Predict the most likely outcomes..."

Output (Ground Truth):
  [
    {"event": "Stock price drops 15% in initial trading", "probability": 0.25, "timeframe_months": 3},
    {"event": "CEO announces restructuring plan", "probability": 0.25, "timeframe_months": 1},
    {"event": "Company returns to profitability", "probability": 0.25, "timeframe_months": 1},
    {"event": "Announces modest hiring for AI roles", "probability": 0.25, "timeframe_months": 1}
  ]
```

### What the Model Learned
- Loss: 3.8716 (baseline would be ~8-10)
- The model learned to:
  1. Predict events in the outcome chains
  2. Assign probabilities (currently uniform 0.25 each)
  3. Estimate timeframes
  4. Format as JSON

### Limitations of This Run
1. **Only 10 cases** (need 100+ for production)
2. **Only 1 epoch** (need 3-5 for convergence)
3. **LoRA rank 8** (should be 64 for SFT learning capacity)
4. **Uniform probabilities** (simplified - not realistic)
5. **Template-based data** (lacks real-world complexity)
6. **No evaluation** (can't measure improvement yet)

---

## Production Changes Required

### 1. Real Historical Data Collection

**Current:** Synthetic templates
**Production:** Real historical events with verified outcomes

**Data Sources:**
- Academic papers with case studies
- Economic databases (World Bank, Fed data)
- Geopolitical archives (Council on Foreign Relations)
- Policy analysis (Brookings, RAND, NBER)
- News archives with multi-year follow-ups

**Example Real Case:**
```json
{
  "case_id": "brexit_2016",
  "seed_event": "Brexit vote passes with 52% leave, 48% remain",
  "seed_date": "2016-06-23",
  "context": "Unexpected result, polls predicted remain victory, David Cameron resigns",
  "domain": "Geopolitics",
  "outcome_chain": [
    {
      "depth": 1,
      "event": "Pound sterling drops 10% to 31-year low",
      "date": "2016-06-24",
      "timeframe_months": 0,
      "probability_in_retrospect": 0.85  // Expert estimates of what seemed likely at the time
    },
    {
      "depth": 2,
      "event": "Article 50 triggered by Theresa May",
      "date": "2017-03-29",
      "timeframe_months": 9,
      "probability_in_retrospect": 0.65
    },
    {
      "depth": 3,
      "event": "Theresa May resigns after failed Brexit negotiations",
      "date": "2019-07-24",
      "timeframe_months": 28,
      "probability_in_retrospect": 0.30
    }
  ],
  "sources": [
    "https://www.brookings.edu/articles/brexit-a-year-later/",
    "https://www.cfr.org/timeline/united-kingdoms-brexit-saga"
  ]
}
```

**Collection Strategy:**
1. **Automated approach:**
   - Use LLM to extract from academic papers
   - Verify with fact-checking APIs
   - Cross-reference multiple sources

2. **Manual curation:**
   - Expert annotation of probabilities
   - Verification of outcome chains
   - Quality control

**Target:** 100-500 real historical cases

---

### 2. Improved Probability Labels

**Current:** Uniform probabilities (all outcomes weighted equally)

**Production:** Retrospective probability estimates

**Methods:**
1. **Expert judgment:** Survey domain experts on "what seemed likely at the time"
2. **Market-based:** Use prediction market data (Polymarket, PredictIt historical)
3. **Survey data:** Polls, economic forecasts from the time
4. **LLM estimation:** Use Claude/GPT-4 to estimate probabilities based on historical context

**Example:**
```json
{
  "depth": 1,
  "event": "Article 50 triggered",
  "actual_probability": 0.65,  // What seemed likely at the time
  "alternative_outcomes": [
    {"event": "Second referendum called", "probability": 0.20},
    {"event": "Brexit cancelled after political crisis", "probability": 0.10},
    {"event": "Indefinite delay announced", "probability": 0.05}
  ]
}
```

---

### 3. Training Configuration Changes

| Parameter | Test Run | Production | Reason |
|-----------|----------|------------|--------|
| **Num cases** | 10 | 100-500 | Better generalization |
| **Epochs** | 1 | 3-5 | Convergence |
| **LoRA rank** | 8 | 64 | SFT needs capacity |
| **Batch size** | 1 | 1-2 | Memory constraints |
| **Learning rate** | 3e-4 | 3e-4 | Keep same |
| **Duration** | 2.3 min | 2-4 hours | 100x data, 3x epochs |
| **Cost** | $0.04 | $6-9 | A10G @ $0.50/hr |

**Production command:**
```bash
python3 -m modal run training/modal_sft.py \
  --num-epochs=3 \
  --lora-rank=64 \
  --data-path=/data/historical_cases.jsonl
```

---

### 4. Evaluation Pipeline

**Current:** No evaluation (just trained successfully)

**Production:** Comprehensive metrics

**Metrics to Track:**
1. **Loss & Perplexity**
   - Baseline: ~8.5
   - Target after SFT: <2.8
   - Target after GRPO: <1.9

2. **Brier Score** (calibration)
   - Measures (predicted_prob - actual_outcome)²
   - Target: <0.15

3. **Match Coverage**
   - How well predictions align with actual events
   - Exact matches vs semantic vs LLM judge
   - Target: >80% match rate

4. **Per-Depth Performance**
   - Depth 1 (immediate): >90% accuracy
   - Depth 2-3 (medium-term): >70% accuracy
   - Depth 4-5 (long-term): >50% accuracy

**Implementation:**
```bash
# Run evaluation
python3 training/evaluation/evaluator.py

# Compare baseline vs SFT
python3 -c "
from training.inference import ProbabilityTreeInference
from training.evaluation.evaluator import TreeEvaluator

# Baseline
inference = ProbabilityTreeInference()
inference.load_adapter(None)
baseline_tree = inference.generate_tree('Brexit vote passes', '52% leave')

# SFT
inference.load_adapter('/data/models/sft/final', 'sft')
sft_tree = inference.generate_tree('Brexit vote passes', '52% leave')

# Evaluate
evaluator = TreeEvaluator()
print('Baseline metrics:', evaluator.evaluate(baseline_tree, ground_truth))
print('SFT metrics:', evaluator.evaluate(sft_tree, ground_truth))
"
```

---

### 5. Data Quality Requirements

**Synthetic Data Issues:**
- ❌ All outcomes have uniform probabilities (0.25 each)
- ❌ Simplified causal chains (4 events each)
- ❌ Missing alternative branches (only shows what happened)
- ❌ No probability uncertainty

**Production Data Requirements:**
- ✅ **Branching outcomes:** Include events that DIDN'T happen
- ✅ **Calibrated probabilities:** Based on expert judgment or market data
- ✅ **Rich context:** Economic indicators, polling data, expert quotes
- ✅ **Multiple depths:** 1-5 levels of consequence chains
- ✅ **Verified sources:** Links to academic papers, archives, databases

**Example Production Case:**
```json
{
  "case_id": "fed_rate_hike_2022_09",
  "seed_event": "Federal Reserve raises interest rates by 75 basis points",
  "seed_date": "2022-09-21",
  "context": {
    "inflation": "8.2% (40-year high)",
    "unemployment": "3.7%",
    "gdp_growth": "0.6%",
    "previous_hikes": "4 consecutive 75bp hikes",
    "market_expectations": "63% expected 75bp, 37% expected 100bp (CME FedWatch)"
  },
  "domain": "Economics",
  "outcome_chain": [
    {
      "depth": 1,
      "event": "30-year mortgage rates cross 7% threshold",
      "date": "2022-10-27",
      "timeframe_months": 1,
      "probability_retrospective": 0.75,
      "alternatives": [
        {"event": "Rates stabilize at 6.5-7%", "prob": 0.20},
        {"event": "Fed signals pause, rates drop", "prob": 0.05}
      ],
      "sources": [
        "https://fred.stlouisfed.org/series/MORTGAGE30US",
        "https://www.brookings.edu/articles/2022-monetary-policy-review/"
      ]
    },
    {
      "depth": 2,
      "event": "Housing sales decline 25% year-over-year",
      "date": "2022-11-22",
      "timeframe_months": 2,
      "probability_retrospective": 0.60,
      "alternatives": [
        {"event": "Sales decline stabilizes at 15%", "prob": 0.25},
        {"event": "Cash buyers sustain market", "prob": 0.15}
      ]
    }
  ]
}
```

---

### 6. Model Architecture Decisions

**Current Setup:**
- Base: GPT-OSS-20B (21B params, 3.6B active MoE)
- LoRA rank: 8 (test) → 64 (production)
- Trainable params: 3.98M (0.02% of total)

**Production Considerations:**

**Option A: Scale up LoRA rank (Current Plan)**
- LoRA rank 64 → 31.85M trainable params
- Same GPU, slightly more memory
- Cost: ~$6-9 for full training

**Option B: Full Fine-Tuning (NOT RECOMMENDED)**
- All 21B params trainable
- Requires 8x H100s, ~$100+/hr
- Cost: $500-1000 for training
- Benefit: Marginal vs LoRA

**Option C: Smaller Model (Fallback)**
- Use 7B model (Mistral, Llama) instead
- Faster, cheaper, but less capable
- Cost: ~$1-2 for training

**Recommendation:** Stick with **Option A** (LoRA rank 64)

---

### 7. Phase 2: GRPO Training

**After SFT completes, add reinforcement learning:**

**Purpose:** Multi-objective optimization
1. **Calibration:** Probabilities match real frequencies
2. **Sharpness:** Confident predictions (not hedged)
3. **Diversity:** Multiple plausible branches
4. **Coherence:** Logical causal chains

**Configuration:**
```python
# modal_grpo.py (already exists)
lora_rank=4  # Ultra-low for RL (per Thinking Machines paper)
gpu="A10G"   # Even cheaper than SFT!
learning_rate=5e-4
num_epochs=3-4
cost=~$2-3
```

**Timeline:**
1. SFT training: 2-4 hours → Perplexity ~2.8
2. GRPO training: 3-4 hours → Perplexity ~1.9
3. Total: ~8 hours, $8-12

---

### 8. Production Pipeline Workflow

```mermaid
1. Data Collection (Manual + Automated)
   ├─ Research historical events (2018-2023)
   ├─ Extract outcome chains
   ├─ Estimate retrospective probabilities
   ├─ Verify with multiple sources
   └─ Format as JSONL

2. Data Validation
   ├─ Check probability sums = 1.0
   ├─ Verify dates are chronological
   ├─ Validate source URLs
   └─ Remove duplicates

3. SFT Training (Phase 1)
   ├─ Upload data to Modal volume
   ├─ Train with LoRA rank 64
   ├─ Save checkpoints every 50 steps
   └─ Monitor loss convergence

4. Evaluation
   ├─ Test on held-out cases
   ├─ Calculate perplexity, Brier score
   ├─ Check match coverage
   └─ Compare vs baseline

5. GRPO Training (Phase 2)
   ├─ Load SFT checkpoint
   ├─ Train with rank 4 LoRA
   ├─ Optimize multi-objective reward
   └─ Save final model

6. Deployment
   ├─ Create Modal inference endpoint
   ├─ Add caching layer
   ├─ Integrate with frontend
   └─ Monitor live predictions
```

---

## Immediate Next Steps (Priority Order)

### Step 1: Full SFT Training on Synthetic Data ✅ READY
**Why:** Verify training converges with more epochs/higher rank

```bash
python3 -m modal run training/modal_sft.py \
  --num-epochs=3 \
  --lora-rank=64
```

**Expected:**
- Duration: ~10-15 minutes (10 cases × 3 epochs)
- Loss: 3.8 → ~1.5 (should overfit on 10 cases)
- Cost: ~$0.15
- Validates: Training loop works end-to-end

---

### Step 2: Collect Real Historical Data 🔴 CRITICAL PATH
**Why:** Can't have production model without real data

**Data Collection Agents (To Build):**

**A. Brainstormer Agent**
- Input: Time period (e.g., "2018-2022")
- Output: List of significant historical events
- Method: Search news archives, academic databases

**B. Chronicler Agent**
- Input: Seed event from Brainstormer
- Output: Complete outcome chain with dates
- Method: Time-ordered search, fact verification

**C. Estimator Agent**
- Input: Seed + outcome chain
- Output: Retrospective probability estimates
- Method: LLM + expert judgment synthesis

**Implementation:**
```bash
# Build data collection pipeline
python3 training/scripts/collect_historical_data.py \
  --start-date=2018-01-01 \
  --end-date=2022-12-31 \
  --num-cases=100 \
  --domains=economics,geopolitics,technology

# This will create: training/data/historical_cases.jsonl
```

**Timeline:** 1-2 weeks to collect 100 quality cases

---

### Step 3: Train on Real Data
**Why:** Get production-ready model

```bash
# Upload real data
python3 training/test_volume.py --upload=historical_cases.jsonl

# Train SFT
python3 -m modal run training/modal_sft.py \
  --num-epochs=3 \
  --lora-rank=64 \
  --data-path=/data/historical_cases.jsonl

# Expected: 2-4 hours, $6-9
```

---

### Step 4: Evaluation
**Why:** Measure if training actually improved predictions

```python
# Run evaluator
python3 training/evaluation/evaluator.py \
  --model-path=/data/models/sft/final \
  --test-data=training/data/historical_cases_test.jsonl

# Expected metrics:
# - Baseline perplexity: 8.5
# - SFT perplexity: 2.8
# - Improvement: 67% reduction
```

---

### Step 5: GRPO Phase 2 (Optional)
**Why:** Further improve calibration and sharpness

```bash
python3 -m modal run training/modal_grpo.py \
  --sft-checkpoint=/data/models/sft/final \
  --lora-rank=4 \
  --num-epochs=3

# Expected: 3-4 hours, $2-3
```

---

### Step 6: Deploy Inference Endpoint
**Why:** Make model accessible to frontend

```python
# Create Modal web endpoint
@app.function(gpu="A10G", keep_warm=1)
@modal.web_endpoint(method="POST")
def generate_tree(request: dict):
    inference = ProbabilityTreeInference()
    inference.load_adapter("/data/models/grpo/final")
    return inference.generate_tree(
        seed_event=request["event"],
        context=request["context"]
    )
```

**Frontend Integration:**
```typescript
// Update src/lib/llm/llm-client.ts
const response = await fetch('https://yourapp.modal.run/generate_tree', {
  method: 'POST',
  body: JSON.stringify({ event, context })
});
```

---

## Cost Projections

### Development (Synthetic Data - Current)
- ✅ 1-epoch test: $0.04
- Full SFT (3 epochs, rank 64): ~$0.15
- **Total: $0.19**

### Production (100 Real Cases)
- Data collection (LLM API calls): ~$50
- SFT training (3 epochs, rank 64): ~$6-9
- GRPO training (3 epochs, rank 4): ~$2-3
- Evaluation runs: ~$2
- **Total: ~$60-64**

### Ongoing Inference
- Per tree generation: ~$0.02
- 1000 trees/day: ~$20/day
- With caching (80% hit rate): ~$4/day
- Monthly (cached): ~$120/month

---

## Success Criteria

### Training Success
- [x] Pipeline runs without errors ✅
- [ ] Perplexity < 2.8 (SFT)
- [ ] Perplexity < 1.9 (GRPO)
- [ ] Match rate > 80%
- [ ] Brier score < 0.15

### Data Quality
- [ ] 100+ real historical cases
- [ ] All sources verified
- [ ] Probabilities sum to 1.0
- [ ] Outcome chains 3-5 events deep
- [ ] Multiple domains covered

### Production Readiness
- [ ] Inference endpoint deployed
- [ ] Frontend integrated
- [ ] Caching layer implemented
- [ ] Monitoring dashboard
- [ ] A/B test (baseline vs trained)

---

## Timeline to Production

| Phase | Duration | Cost | Status |
|-------|----------|------|--------|
| ✅ Pipeline dev | 1 week | $1 | Complete |
| 🟡 Data collection | 2 weeks | $50 | Not started |
| 🟢 SFT training | 4 hours | $9 | Ready to run |
| 🟢 GRPO training | 4 hours | $3 | Ready to run |
| 🟢 Evaluation | 1 day | $2 | Ready to run |
| 🟢 Deployment | 2 days | $0 | Ready to run |
| **Total** | **3-4 weeks** | **~$65** | **Blocked on data** |

**Critical Path:** Real historical data collection

---

## Quick Wins (Can Do Now)

1. **Run full SFT on synthetic** (validate 64-rank convergence)
2. **Test evaluation pipeline** (ensure metrics work)
3. **Build data collection agents** (Brainstormer/Chronicler)
4. **Create 10 real cases manually** (validate data format)
5. **Set up inference endpoint** (prepare for deployment)

---

## Key Decisions Needed

### 1. Data Collection Strategy
- [ ] Automated (faster, lower quality)?
- [ ] Manual (slower, higher quality)?
- [ ] Hybrid (automated + manual verification)?

### 2. Probability Estimation Method
- [ ] Expert surveys?
- [ ] LLM estimation?
- [ ] Historical prediction markets?
- [ ] Combination?

### 3. Model Size
- [ ] Stick with GPT-OSS-20B (best quality)?
- [ ] Try smaller model (faster inference)?

### 4. Deployment Priority
- [ ] Train first, then deploy?
- [ ] Deploy baseline now, upgrade later?

**Recommendation:**
- Use **hybrid data collection** (LLM extraction + manual verification)
- Use **LLM probability estimation** (cheapest, scalable)
- **Stick with GPT-OSS-20B** (working setup, good performance)
- **Train first** (only takes 8 hours once data is ready)
