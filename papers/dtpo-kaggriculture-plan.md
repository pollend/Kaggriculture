# DTPO for Kaggriculture

## References

- Daniël Vos and Sicco Verwer, [“Optimizing Interpretable Decision Tree Policies for Reinforcement Learning”](https://arxiv.org/html/2408.11632v1), 2024.
- [Official DTPO implementation](https://github.com/tudelft-cda-lab/DTPO).

## Goal

Explore whether a small, interpretable decision tree can replace or complement the neural policies used by the DQN and SP-PSRO agents. The first target should be high-level purchasing, selling, and animal planning. Field execution should remain with the existing job scorer until the simpler experiment establishes that a tree policy can learn useful season-level strategy.

The desired result is a policy whose behavior can be inspected as rules such as:

```text
if remaining_days <= 3:
    liquidate inventory
elif wheat_reserve < expected_animal_feed:
    buy or retain wheat
elif pasture_capacity > animals_owned and money > safety_reserve:
    buy animal
else:
    hold cash
```

These are illustrative rules only. The learned tree, rather than hand-written thresholds, must determine the final splits and leaf actions.

## What DTPO Does

Decision Tree Policy Optimization (DTPO) directly trains a crisp decision-tree policy with policy-gradient information. It does not require a neural policy teacher.

At a high level, DTPO:

1. Starts with a single-leaf tree that assigns equal probability to each action.
2. Collects a large on-policy rollout batch.
3. Uses a neural value function and generalized advantage estimation to estimate which sampled actions were beneficial.
4. Converts the policy-gradient update into new probability targets.
5. Fits a complete multi-output regression tree to those targets.
6. Accepts the replacement tree only when it improves the policy loss.
7. Periodically converts the tree to a deterministic policy and retains the best evaluated version.

The neural critic exists only during training; inference uses the decision tree. The paper commonly limits policies to 16 leaves for interpretability, although its results show that harder environments may need substantially larger trees. Its standard experiments use 10,000 environment steps per tree update, emphasizing policy quality rather than sample efficiency.

## Fit with the Current Agents

The paper assumes a fixed discrete action set and learns a vector of action probabilities from each state. Kaggriculture’s current learned decisions do not have that exact shape.

### Field jobs

The DQN job scorer dynamically enumerates up to 96 legal jobs. It evaluates a feature vector for each `(state, candidate job)` pair and selects among currently allowed candidates. Jobs are assigned sequentially to the farmer and hired hands, with commitments and masks preventing conflicting assignments.

A paper-faithful tree that predicts one fixed action-probability vector therefore cannot directly replace the field network. A field-tree experiment would need an adapted contextual policy:

```text
tree(state, candidate) -> candidate logit
softmax(logits of currently legal candidates) -> selection probabilities
```

This preserves the dynamic action set but is a research adaptation of DTPO, not the algorithm exactly as evaluated in the paper.

### Market and animal planning

The market policy makes a smaller set of high-level choices once per day: a purchase plan followed by per-product reserve decisions. Its observation already includes money, shed and seed inventory, farm composition, market supply and prices, town demand, season progress, and animal-related state.

This is the best initial target because a compact fixed catalog can represent strategic choices while the deterministic market executor continues to enforce legality, order limits, feed survival, cash reserves, and endgame liquidation.

Animal planning must cover more than buying livestock. The state and actions must account for:

- existing coops and pastures;
- animals owned, stored, and already placed;
- wheat needed to feed animals through the planning horizon;
- farmer and hand capacity for feeding, care, fertilizer collection, and product collection;
- current animal products and their market demand;
- remaining days, because animals need enough production cycles to recover their purchase and structure costs.

## Candidate Approaches

### 1. DTPO high-level planner — recommended first

Train a decision tree over a fixed set of daily portfolio actions, for example:

- hold the current plan;
- emphasize a crop family;
- build toward geese, cows, or sheep;
- replenish wheat/feed reserves;
- hire or expand land;
- preserve product inventory for expected demand;
- sell normally;
- liquidate for the endgame.

The chosen leaf action becomes a structured plan consumed by the existing safe market executor and job generator. This isolates strategic learning from pathfinding and action legality.

Start with a maximum of 16 leaves. Benchmark 32 and 64 leaves only if 16 leaves materially underperform and the larger trees remain understandable.

### 2. DQN-to-tree extraction baseline

Use the current trained DQN/SP-PSRO policy as a teacher and fit a tree from states, legal candidates, chosen actions, and teacher Q-values. This is closer to VIPER or policy distillation than DTPO.

This baseline is important because the paper reports that imitation often performs well when the DQN teacher is strong. It also provides a lower-cost way to test whether the current features contain clean, tree-friendly decision boundaries.

### 3. Contextual DTPO field scorer

Adapt DTPO so a scalar tree scores each dynamic job candidate. Normalize the legal candidate scores with a softmax, sample one job, commit it, and repeat for the next unit.

This should be attempted only after the market experiment because it introduces variable action sets, sequential within-turn decisions, masks, sticky multi-turn commitments, and shared credit among workers. Results must be described as an adaptation rather than a reproduction of DTPO.

## Proposed Experiment

### Phase 1: fixed-action market and animal tree

1. Define a compact daily state from the existing market features, retaining human-readable feature names and physical units where possible.
2. Define a fixed catalog of mutually exclusive portfolio actions. Keep quantities and legal order construction in the deterministic executor.
3. Create a separate training-capable agent family under `agents/` with its own `agent.toml`; do not overwrite the DQN or SP-PSRO families.
4. Collect on-policy episodes and compute rewards at the daily decision boundary. Include terminal bank value and intermediate potential changes, but avoid assigning terminal value to unsold inventory.
5. Train the DTPO tree with a neural critic. Begin with a 16-leaf limit and paper-aligned values of `gamma=0.99` and `lambda=0.95`; tune rollout size to full Kaggriculture episodes rather than assuming the paper’s 10,000-step batch is sufficient.
6. Export the deterministic tree as primitive arrays or generated Python logic so submission-time inference requires NumPy at most, not scikit-learn or JAX.

### Phase 2: extraction baseline

1. Run the selected DQN/SP-PSRO champion to collect states, legal plans, selected plans, and Q-values.
2. Fit 16-, 32-, and 64-leaf trees using advantage- or regret-weighted samples.
3. Evaluate both teacher fidelity and actual game return; agreement alone is not evidence that the extracted policy is competitive.

### Phase 3: field-policy research

Proceed only if a tree matches the market policy closely enough to justify the additional complexity. Compare direct contextual DTPO with a tree regressor distilled from the DQN job scorer.

## Evaluation

Every tree candidate should be compared with the current joint DQN and deterministic heuristic market baselines using identical seeds and opponent rosters.

Record:

- mean final bank and win rate;
- lower-tail performance and variance across seeds;
- invalid or dropped market orders;
- animal starvation and missed care/collection opportunities;
- shed overflow and unsold terminal inventory;
- decision-tree leaf count and maximum depth;
- number of distinct features used by the tree;
- inference latency and serialized submission size;
- training episodes, wall-clock time, peak RAM, and artifact size;
- fidelity to the DQN for the extraction baseline.

A promising first result is a 16-leaf market/animal tree that retains most of the joint DQN’s return, produces legible rules, never bypasses the safety executor, and fits comfortably within the Kaggle resource limits. Promotion into SP-PSRO should be a later experiment, not part of the initial proof of concept.

## Risks and Open Questions

- Kaggriculture has sparse, delayed rewards; daily actions may not reveal their value until much later in the season.
- Self-play makes the data distribution non-stationary, while DTPO is on-policy and depends on large representative batches.
- A small tree may lack capacity for crop, animal, market, town, opponent, and endgame regimes simultaneously.
- Leaf count alone does not guarantee interpretability if the tree relies on opaque normalized features. Exported rules should use descriptive names and decoded thresholds.
- A fixed high-level action catalog introduces expert structure. That improves tractability but means the tree cannot discover strategies outside the catalog.
- The DQN’s dynamic candidate formulation may make extraction substantially easier than direct DTPO for field jobs.

## Decision

Begin with a new, separate DTPO research agent that controls only the daily high-level market and animal plan. Retain the existing job-scoring network and deterministic safety executor. In parallel, treat DQN-to-tree extraction as the primary baseline. Do not replace the joint DQN, change SP-PSRO training, or attempt a full field-policy tree until the initial comparison demonstrates that an interpretable tree can make competitive portfolio decisions.
