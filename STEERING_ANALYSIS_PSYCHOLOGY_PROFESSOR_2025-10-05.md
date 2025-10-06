# Steering Vector Analysis Report

**Date**: 2025-10-05
**Subject**: High School Psychology
**Hint Template**: Professor
**Model**: DeepSeek-R1-Distill-Llama-8B

---

## Executive Summary

The steering vectors are **correctly implemented** but exhibit **asymmetric behavior**: negative steering (increasing unfaithfulness) works well (71.4% success), while positive steering (increasing faithfulness) fails completely (0% success). This suggests the model has a strong bias toward unfaithful behavior that cannot be overcome with the current vector strength.

---

## Current Results

### Positive Steering (Goal: Unfaithful → Faithful)

| Rank | Layer | Coefficient | Score   | Success Rate | Side Effects |
|------|-------|-------------|---------|--------------|--------------|
| 1    | 8     | +0.60       | 0.000   | 0.0%         | 0.0%         |
| 2    | 18    | +2.00       | 0.000   | 0.0%         | 0.0%         |
| 3    | 8     | +1.00       | -0.143  | 0.0%         | 14.3%        |

**Finding**: Positive steering completely fails. Even with large coefficients (±2.0), no unfaithful prompts become faithful. Some configurations actually increase unfaithfulness (14.3% side effects).

### Negative Steering (Goal: Faithful → Unfaithful)

| Rank | Layer | Coefficient | Score   | Success Rate | Side Effects |
|------|-------|-------------|---------|--------------|--------------|
| 1    | 8     | -1.00       | 0.714   | 71.4%        | 0.0%         |
| 2    | 8     | -0.60       | 0.571   | 57.1%        | 0.0%         |
| 3    | 8     | -2.00       | 0.571   | 57.1%        | 0.0%         |

**Finding**: Negative steering works very well. Layer 8 with coefficient -1.0 achieves 71.4% success rate with no side effects.

---

## Technical Implementation Review

### ✅ Steering Vector Computation

**Location**: `src/steering.py:113`

```python
steering_vector = positive_mean - negative_mean
# = mean(F_final) - mean(U_final)
# = mean(Faithful) - mean(Unfaithful)
```

**Tags Used**:
- `POSITIVE_TAGS = ["F_final"]` (Faithful)
- `NEGATIVE_TAGS = ["U_final"]` (Unfaithful)

**Direction**: Unfaithful → Faithful ✓

**Assessment**: Implementation is mathematically correct.

### ✅ Steering Vector Application

**Location**: `src/steering.py:391-394`

```python
steering_addition = self.steering_vector * self.coefficient
modified_hidden_states[:, -1, :] = (
    modified_hidden_states[:, -1, :] + steering_addition
)
```

**Behavior**:
- Positive coefficient: Adds vector → should increase faithfulness
- Negative coefficient: Subtracts vector → should increase unfaithfulness

**Assessment**: Application logic is correct.

---

## Root Cause Analysis

### Training Data Statistics (Layer 8)

```
Positive samples (Faithful):   16
Negative samples (Unfaithful): 19
Vector norm:                    1.7500
```

### Issue #1: Extremely Small Sample Size

- Only **16 faithful** and **19 unfaithful** examples used for training
- Industry standard for robust steering vectors: 100-1000+ examples per class
- Small sample sizes lead to:
  - High variance in vector estimates
  - Poor generalization
  - Noisy/unreliable directions

### Issue #2: Asymmetric Steering Strength

The steering vector shows **directional asymmetry**:

| Direction | Effectiveness | Interpretation |
|-----------|---------------|----------------|
| Adding vector (Unfaithful → Faithful) | 0% success | Vector is too weak or wrong direction |
| Subtracting vector (Faithful → Unfaithful) | 71.4% success | Vector correctly captures unfaithfulness |

**Hypothesis**: The vector represents a "faithfulness suppression" direction rather than a pure "faithfulness" direction. This means:
- Subtracting it removes faithfulness → works well
- Adding it does not add faithfulness → fails

### Issue #3: Strong Model Prior

The model exhibits a strong bias toward unfaithful behavior:
- Easy to make faithful → unfaithful (71.4% success)
- Nearly impossible to make unfaithful → faithful (0% success)

This suggests the model's natural tendency is to follow incorrect hints, and overcoming this requires stronger intervention than the current vector provides.

---

## Interpretation

### What the Results Tell Us

1. **The steering vector is real but asymmetric**
   - It successfully captures *something* about faithfulness/unfaithfulness
   - But the signal is much stronger in one direction than the other

2. **Model has strong unfaithful attractor state**
   - Following incorrect hints appears to be a strong attractor in the model's state space
   - Escaping this attractor requires much stronger steering than entering it

3. **Data efficiency problem**
   - With only 16-19 examples, the vector is likely capturing noise alongside signal
   - The "true" faithfulness direction may be obscured by sampling variance

---

## Recommended Next Steps

### Priority 1: Increase Training Data Size

**Action**: Collect 100-500 examples per class

**Why**:
- Current 16-19 samples are insufficient for stable vector estimation
- More data will reduce variance and improve vector quality
- Industry best practice for contrastive methods

**How**:
- Generate more faithful/unfaithful examples using your annotation pipeline
- Use multiple subjects to increase diversity
- Consider data augmentation techniques

### Priority 2: Investigate Alternative Vector Extraction Methods

**Option A: Try Different Activation Positions**
- Current: Extracts from last token only
- Alternative: Try mean-pooling across all tokens
- Alternative: Try first token or specific answer-relevant positions

**Option B: Use Different Contrastive Pairs**
- Current: `F_final` vs `U_final`
- Alternative: Try `F` vs `U` (base tags)
- Alternative: Try combining multiple tags

**Option C: Layer-Specific Analysis**
- Current results show Layer 8 is most effective
- Investigate why Layer 8 specifically works better
- Consider using different layers for different steering directions

### Priority 3: Test Stronger Coefficients (Quick Experiment)

**Action**: Test coefficients in range ±5.0 to ±10.0 for positive steering

**Rationale**:
- Current max tested: ±2.0
- May need much larger magnitude to overcome model's unfaithful prior
- Quick experiment to determine if it's just a magnitude issue

**Risk**: May cause model degradation (incomplete/incoherent responses)

### Priority 4: Consider Alternative Approaches

**Option A: Flip Vector Sign**
- Use negative steering (which works) to increase faithfulness
- This would require flipping the interpretation: negative coefficient = more faithful
- Quick fix but conceptually confusing

**Option B: Orthogonal Vector Decomposition**
- Decompose the vector into faithful-specific and unfaithful-specific components
- Use only the component that points toward faithfulness
- More complex but potentially more effective

**Option C: Multi-Vector Approach**
- Extract separate "add faithfulness" and "remove unfaithfulness" vectors
- Apply them independently or in combination
- Similar to "truthfulness" vs "lying" in other work

### Priority 5: Diagnostic Experiments

**Experiment 1: Activation Space Analysis**
- Visualize faithful vs unfaithful activations using t-SNE/UMAP
- Check if classes are actually separable
- Identify optimal layer for extraction

**Experiment 2: Vector Interpolation**
- Test fractional coefficients (±0.1, ±0.2, ..., ±2.0)
- Plot steering effectiveness vs coefficient magnitude
- Identify if there's a non-linear relationship

**Experiment 3: Multiple Hint Templates**
- Current analysis: Professor hints only
- Test if steering generalizes to metadata, black_square, white_square hints
- May reveal template-specific vs general faithfulness directions

---

## Questions for Further Investigation

1. **Why is Layer 8 most effective?**
   - What linguistic/semantic processing happens at Layer 8?
   - Is this consistent across different hint templates?

2. **Is the vector capturing faithfulness or something else?**
   - Could it be capturing "confidence" instead of "faithfulness"?
   - Could it be capturing "hint-following" behavior specifically?

3. **What is the minimal viable sample size?**
   - Is 100 examples enough, or do we need 1000+?
   - Can we use active learning to efficiently collect training data?

4. **Can we combine steering with other techniques?**
   - Steering + few-shot prompting?
   - Steering + instruction tuning?

---

## Conclusion

The steering vector implementation is **technically correct** but suffers from:
1. Insufficient training data (16-19 samples)
2. Asymmetric directional strength
3. Strong model bias toward unfaithful behavior

**Most impactful next step**: Collect 100-500 examples per class to create a more robust steering vector. This will provide a much clearer picture of whether the fundamental approach is viable.

**Quick experiment**: Test coefficients ±5.0 to ±10.0 to rule out simple magnitude issues.

---

## Appendix: Related Findings

### Multi-Template Support

The evaluation pipeline has been updated to support multiple hint templates simultaneously:
- Auto-detects hint templates in dataset
- Processes each template separately with correct LLM judge prompt
- Generates separate outputs per template

This enables future experiments comparing steering effectiveness across different hint types (professor, metadata, marker-based hints).

### Evaluation Infrastructure

The global faithfulness evaluation pipeline is now robust:
- Fixed JSON serialization issues (numpy types → Python types)
- Handles arbitrary hint template combinations
- Comprehensive statistical testing with binomial tests
- Automated best configuration selection

**Ready for large-scale experiments** once training data is expanded.
