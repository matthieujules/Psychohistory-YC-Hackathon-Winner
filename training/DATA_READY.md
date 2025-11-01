# Training Data - READY ✅

**File:** `training/data/curated_cases.jsonl`
**Size:** 292 KB
**Status:** Production Ready

---

## Dataset Summary

### Complete Coverage
- ✅ **91 curated cases** (all 91 verified seeds processed)
- ✅ **273 training examples** (91 cases × 3 depth levels)
- ✅ **1,092 total candidates** (273 actual + 819 alternatives)
- ✅ **0 generic templates** - all outcomes researched and specific

### Split
- **80 post-cutoff** (88%) - Jul 2024 - Mar 2025
  - Outside GPT-OSS-20B training data (June 2024 cutoff)
  - True forecasting test
- **11 in-distribution** (12%) - 2019-2022
  - Within training data  
  - Calibration baseline

### Quality Metrics
- ✅ All cases have exactly 3 levels
- ✅ All levels have 4 candidates (1 actual + 3 alternatives)
- ✅ All outcomes specific, dated, researched
- ✅ All alternatives realistic and plausible
- ✅ No duplicates, all IDs unique (001-091)

---

## Sample Case

**Biden Withdrawal (2024-07-21)**

Month 1: Harris secures nomination with 99% delegate support (Aug 5)
- Alt: Hillary enters race / Newsom challenges / DNC delays

Month 2: Harris selects Tim Walz as VP (Aug 6)  
- Alt: Shapiro for PA / Buttigieg for youth / Convention deadlocks

Month 3: DNC unifies party behind Harris-Walz (Aug 19)
- Alt: Protest disruption / Biden boycott / Progressive walkout

---

## Next Steps

### 1. Upload to Modal Volume
```bash
modal volume put psychohistory-data training/data/curated_cases.jsonl /data/
```

### 2. Train SFT Model
```bash
python3 -m modal run training/modal_sft.py --num-epochs=1
```

Expected:
- Duration: ~30-45 min on H100
- Cost: ~$2-3
- Baseline perplexity: ~8.5
- Target perplexity: ~2.5-3.5

---

## Domain Distribution
- Geopolitics: 30 (33%)
- Technology: 19 (21%)
- Politics: 16 (18%)
- Business: 15 (17%)
- Economics: 11 (12%)

---

**Status:** ✅ READY TO TRAIN
