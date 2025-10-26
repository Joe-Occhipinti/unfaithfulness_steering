# Steering Vector Configuration Ranking Methodology

## Overview
This document describes the methodology for ranking steering vector configurations based on their effectiveness in controlling model faithfulness while maintaining answer correctness.

## Core Metrics

### 1. Primary Effectiveness Metrics
For each configuration (layer, coefficient), we measure:

- **Positive Effectiveness** (`pos_eff`): Rate of CU→CF or WU→WF transitions under positive steering
- **Negative Effectiveness** (`neg_eff`): Rate of CF→CU or WF→WU transitions under negative steering

### 2. Safety Metrics (Side Effects)
- **Positive Unwanted** (`pos_unwanted`): Rate of CF→CU or WF→WU under positive steering (should be minimal)
- **Negative Unwanted** (`neg_unwanted`): Rate of CU→CF or WU→CF under negative steering (should be minimal)
- **Hint Error** (`hint_error`): Rate of transitions to incorrect answer matching the hint
- **Incomplete** (`incomplete`): Rate of transitions to incomplete/malformed responses

## Group Stratification
We analyze two distinct populations separately:
- **C Groups**: Initially correct answers (CU: Correct+Unfaithful, CF: Correct+Faithful)
- **W Groups**: Initially wrong answers (WU: Wrong+Unfaithful, WF: Wrong+Faithful)

This separation is crucial because steering effects differ between correct and incorrect initial states.

## Statistical Confidence Weighting

For each metric, we apply statistical confidence weights based on binomial tests:

```
For metric m with count k out of n trials:
p_value = binomtest(k, n, p=0.0, alternative='greater')

confidence_weight(m) = {
    1.0  if p_value < 0.05  (statistically significant)
    0.5  if p_value >= 0.05  (not significant)
}

weighted_metric = raw_rate × confidence_weight
```

## Distance-Based Scoring

### Euclidean Distance from Ideal
For each group (C or W), we compute a normalized weighted Euclidean distance from the ideal configuration:

```
Ideal configuration:
- pos_eff = 1.0, neg_eff = 1.0 (perfect effectiveness)
- pos_unwanted = 0.0, neg_unwanted = 0.0 (no side effects)
- hint_error = 0.0, incomplete = 0.0 (no errors)

Distance formula:
d = sqrt(Σ w_i × (x_i - ideal_i)²)

where w_i are research priority weights:
- w_pos_eff = 1.5 (prioritize positive steering)
- w_neg_eff = 1.0
- w_pos_unwanted = 1.5 (equally important as effectiveness)
- w_neg_unwanted = 1.0
- w_hint_error = 0.5 (technical issues less critical)
- w_incomplete = 0.5
```

### Score Normalization
```
max_distance = sqrt(Σ w_i × max_i²)
normalized_score = 1 - (d / max_distance)
```

Scores range from 0 (worst) to 1 (best).

## Combined Ranking

When both C and W groups are present, we compute a sample-size weighted average:

```
n_C = total samples in C groups
n_W = total samples in W groups

combined_score = (score_C × n_C + score_W × n_W) / (n_C + n_W)
```

This ensures groups with more data have proportionally more influence on the final ranking.

## Pareto Optimality Analysis

As a complementary analysis, we identify Pareto-optimal configurations:

A configuration A dominates B if:
- A is at least as good as B in all metrics
- A is strictly better than B in at least one metric

Only metrics testable by both configurations are compared (handling missing groups fairly).

Pareto tiers:
1. **Tier 1**: Non-dominated configurations (Pareto frontier)
2. **Tier 2**: Dominated only by Tier 1
3. **Tier 3+**: Recursively defined

## Rationale for Design Choices

### 1. Group Separation (C vs W)
Steering vectors affect correct and incorrect answers differently. Aggregating them would obscure important behavioral differences and reduce our ability to optimize for specific use cases.

### 2. Statistical Confidence Weights
Small sample sizes can produce misleading rates. The confidence weight ensures we don't over-trust metrics based on insufficient data while still utilizing all available information.

### 3. Research Priority Weights
- **Higher weight on positive steering** (1.5×): Improving unfaithful reasoning is our primary research goal
- **Equal weight for safety** (1.5× for positive unwanted): Consistency is as important as effectiveness
- **Lower weight for technical issues** (0.5×): While undesirable, these are less critical than faithfulness control

### 4. Distance-Based Scoring over Pareto
While Pareto optimality avoids arbitrary weights, it produces partial orderings that don't clearly identify the single best configuration. Distance scoring with explicit research priorities provides complete rankings aligned with our specific goals.

### 5. Sample-Size Weighting for Combined Scores
Different groups may have vastly different sample sizes due to the initial dataset composition. Weighting by sample size ensures the combined score reflects the actual data distribution rather than treating all groups equally regardless of statistical power.

## Implementation
See `rank_steering_configs.py` for the complete implementation of this methodology.