# dqn_market_free — the animal-product niche, and why the agent is not in it

What each session changed and measured is in
[dqn_market_free_sessions.md](dqn_market_free_sessions.md); the general argument
about what to build next is in
[dqn_market_free_roadmap.md](dqn_market_free_roadmap.md). This file is one
specific unbuilt thing, written up while the evidence was in front of us.

Status: **items 1, 2 and 4 built in `dqn_market_free_v2`**, session
`market-free-herd-v1`; item 3 is the measurement that follows and has not run yet.
See "What was built" at the end. Session `market-free-landheuristic-v43` was
mid-flight (generation 17, batch ~176) when the rest of this was written, so every
league number below is a snapshot of a live run and will have moved.

Where this sits against the ladder overall: [dqn_market_free_ladder_standing.md](dqn_market_free_ladder_standing.md)

## The finding

Kaggle episode `94146866`, seed 1837443377. Jan Moilanen 30,136 vs our
`landheur_02` submission 16,880. Their first turn is:

```json
{"farmer": ["BUILD_COOP"], "hands": [],
 "market": [["HIRE"], ["HIRE"], ["BUY_PRODUCT", "WHEAT", 10],
            ["BUY_LAND"], ["BUY_ANIMAL", "GOOSE", 5]]}
```

They then do almost no farming for the rest of the episode:

| | Jan Moilanen | ours |
| --- | --- | --- |
| geese / coops at end | 16 / 16 | — |
| pastures built | — | 4 |
| animals bought | 17 GOOSE | **0** |
| `SELL EGG` units requested | 418 | 0 |
| `SELL` orders, all items | 49 | 361 |
| `BUY_PRODUCT` orders | 197 | 310 |
| HARVEST | 12 | 25 |
| seed purchases | 7 | 13 |
| BUY_LAND | 1 | — |
| FEED / CARE | 154 / 31 | 0 / 0 |

Their money sits near zero from day 3 to day 18 — every coin goes back into
geese — then runs 103 → 30,136 over the last eleven days. It is a compounding
egg engine with a farm attached for feed, not a farm.

Per-item counts above are *requested* order quantities, not filled ones. For
Jan's 49 sell orders the two are close; for our 361 they are not, and our
`BUY_PRODUCT` traffic partly offsets our own sales. Treat our column as gross
traffic.

## Why the niche is open

Market inventory and price across that episode:

| item | inventory | price | supplied by |
| --- | --- | --- | --- |
| **EGG** | 10000 → 9848 | 50 → 59 (+18%) | Jan only |
| **MILK** | 10000 → 9880 | 160 → 255 (+59%) | nobody |
| **WOOL** | 10000 → 9862 | 200 → 242 (+21%) | nobody |
| WHEAT | 10000 → 8999 | 25 → 57 (+128%) | net demand |
| MELON | 10000 → 10127 | 250 → 89 (−64%) | us |
| FERTILIZER | 10000 → 10101 | 100 → 80 (−20%) | us |

Jan sold 418 eggs and egg inventory still **ended below `I0`** — town demand
outran his entire output, so his unit price rose 18% while he dumped everything
he made. All three animal products end with inventory down and price up.

Our volume goes into the two items whose price collapses. Melon's crash is
self-inflicted: inventory +127 against `T=300, above_func: sq, above_target:
3.60`.

This is the part that reframes the work. It is not "the agent is missing an
option." It is **the agent competes where price punishes volume and is absent
where price rewards it.** The relevant `MARKET_PARAMS` spreads:

| resource | T | above func | above target | P(I0+2T) |
| --- | --- | --- | --- | --- |
| EGG | 332 | log | 0.20 | $39 |
| WHEAT | 400 | log | 0.20 | $19 |
| MILK | 122 | linear | 1.60 | $1 |
| WOOL | 105 | sq | 3.20 | $1 |
| MELON | 300 | sq | 3.60 | $1 |
| STRAWBERRY | 100 | linear | 1.60 | $1 |

EGG is the deepest premium market in the game and the only animal product that
survives a glut. Its per-tile-per-day yield is 1.00 (goose) against 0.50 (cow)
and 0.33 (sheep), at 300 cost against 400 and 500.

## Why the policy is not there

Nothing masks it. `BUY_ANIMAL GOOSE/COW/SHEEP` are action indices 177-185,
`SELL EGG` is 56-66, `SELL WOOL` is 78-88, and `lots.py` attributes animal
credit for all three (`ANIMAL_ITEMS` is `tuple(donor.ANIMALS)`). The comment at
`agents/dqn_market_free/order_space.py:92` records the last time EGG *was*
silently unreachable and why it no longer is.

Three separate causes, in the order they bite.

### 1. Pasture is unconditional; coop is not

`herd.py` assigns COW/SHEEP roles from `donor.CORE_HERD_SEQUENCE` whether or not
an animal is owned, so pasture gets built in **every** episode. GOOSE gets a
role only from `donor._private_item_total(private, GOOSE)`
(`agents/dqn_market_free/herd.py:106-128`) — that is, only *after* a goose has
already been bought.

So the two purchases are not symmetric:

- `BUY_ANIMAL COW` lands in a pasture that already exists and starts paying.
- `BUY_ANIMAL GOOSE` spends 300, then waits on a `BUILD_COOP`, then a separate
  `PLACE` turn. Longer delay to first reward, and a goose that never gets housed
  is −300 with no offset.

Both halves of that asymmetry show up in the league. Sampled at generation ≥15
(7,360 episodes at the time of sampling):

| | episodes | share |
| --- | --- | --- |
| BUILD_PASTURE | 7360 | **100%** |
| BUILD_COOP | 80 | **1.1%** |
| sells MILK | 2688 | 37% |
| sells WOOL | 306 | 4.2% |
| sells EGG | 67 | 0.9% |

80 coops against 67 egg-selling episodes: eggs happen only when a coop happened,
and a coop happens almost never.

The other half is pure waste. Re-sampled later in the same generation range
(9,600 episodes):

- **5,018 episodes (52%) buy zero animals** — and still build **14.6 pastures**
  each on average.
- Those episodes average 30,353 money against 36,059 for episodes that buy at
  least one animal, **−16%**.

Half the time the agent pays for a herd it never buys. The Kaggle replay above
is that case exactly: 4 pastures, 0 animals, 720 steps.

### 2. Exploration cannot find it

199 action columns, three of them goose. Noisy layers are head-only at
`noisy_order_rate_target: 0.05`. Discovering the goose column *and* then
following through a build-and-place sequence is not something 5% head noise
reaches. `self_play_ratio: 0.65` means the opponents are copies of the same
policy, so nothing demonstrates it either.

### 3. Where it has been tried, it currently loses

Generation ≥15, by which animal products an episode sold:

| animal products sold | episodes | mean money | BUY_ANIMAL |
| --- | --- | --- | --- |
| milk only | 2599 | 40,007 | 8.8 |
| none | 4552 | 29,145 | 6.8 |
| egg | 55 | 29,048 | 12.7 |
| wool | 314 | 26,064 | 11.4 |

Wool and egg episodes buy 30-45% more animals and end 25-35% poorer. Confounded
by capital tied up in late-paying assets and by cause 1 above, but the gradient
the learner sees is real, so it will not climb out of this on its own.

Wool is also a genuinely bad fit for how this policy sells. `T=105`,
`above_func: sq`, `above_target: 3.20` is the thinnest market in the game, and
the policy dumps — roughly 320 SELL orders per episode against a quantity ladder
reaching 45. That is right for wheat and wrong for wool.

## Plan

Ordered by evidence strength, not by effort. 1 and 2 are the same module and
should land together; each is independently testable.

1. **Stop building pastures for animals that were never bought.** Gate the
   COW/SHEEP role assignment in `herd.py` on ownership or on a committed
   purchase, the way GOOSE is already gated. This is the 52%-of-episodes,
   14.6-pastures-each waste, and it is a strict improvement independent of
   anything else here — it removes a cost, it does not add a bet.

2. **Pre-assign a coop the way pasture is pre-assigned.** Make
   `BUY_ANIMAL GOOSE` land in an existing coop on arrival, so it is a one-step
   purchase like the cow instead of a three-step gamble. Combined with 1, the
   rule becomes "structures follow committed animals" for all three species
   rather than "pasture always, coop never."

3. **Re-measure before touching exploration.** 1 and 2 change the payoff of the
   goose column without touching the policy. Run a session and check whether
   BUILD_COOP share moves off 1.1% on its own. Only if it does not, consider a
   demonstration or a targeted exploration bonus on the animal columns —
   `demonstration_episodes` is already a config key and is currently 0.

4. **Size the sell ladder to market depth.** Separate from the animal work but
   the same root cause. The quantity ladder tops out at 45 and is applied
   identically to WHEAT (`T=400`, log) and WOOL (`T=105`, sq). A depth-aware cap
   would stop the policy from crushing its own price on the thin premium items,
   which is also what melon's −64% is.

## How to tell whether it worked

Per session, at generation boundaries, filtering `NOT failed` on both `episodes`
and `episode_players`:

- `BUILD_PASTURE` count in episodes with zero `BUY_ANIMAL` → should approach 0.
  This is the check for item 1 and needs no learning to pass.
- Share of episodes with `BUILD_COOP` → currently 1.1%.
- Share of episodes selling EGG → currently 0.9%.
- Mean money in zero-animal episodes → currently 30,353 against 36,059 for
  animal-buying ones; the gap should close from the left.

Rank checkpoints on `challenger_benchmarks[0].a_money_median`, not `lr_score` —
the arena score swings ~48% per update against 11.7% for the rollout mean — and
screen
`behavior.unrecovered_animal_share`. Anything above ~0.1 is the animal blowup
pattern: mean carried by rare episodes, median in the dirt. `landheur_04` at
batch 170 was the worst lr_score-vs-reality case seen — 28,754 lr_score,
benchmark mean 8,177, benchmark median 2,508, `unrecovered_animal_share` 0.55.

Adding animals to a policy that cannot manage them is how that failure mode is
produced, so item 1 landing before item 2 is the point, not an ordering
preference.

## The decay curve, which the aggregates hid

Counted off v1's own rollout traces
(`artifacts/dqn_market_free/dqn_market_free/landheur_0*/episode_traces/**/*.npz`),
60 episodes sampled per generation, `order_chosen` in the `BUY_ANIMAL` columns
177-185 and the `SELL` columns 56-88:

| gen | GOOSE buys /ep | COW | SHEEP | SELL EGG | SELL MILK | turns GOOSE unmasked /ep |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.33 | 3.53 | 7.77 | 3.77 | 0.82 | 162 |
| 5 | 2.68 | 4.23 | 3.10 | 2.25 | 1.50 | 379 |
| 13 | 0.22 | 3.83 | 0.28 | 0.00 | 6.95 | 480 |
| 27 | 0.20 | 7.22 | 1.37 | 0.13 | 14.02 | 502 |
| 42 | 0.00 | 9.12 | 0.67 | 0.00 | 14.43 | 503 |
| 53 | 0.05 | 8.37 | 1.02 | 0.00 | 10.15 | 511 |

Two things in that table change the plan above.

**It is not an exploration failure.** The goose column is open ~500 turns an
episode the whole way, and generation 1 bought geese ten times an episode. The
policy found the action and priced it out. Demonstrations and exploration bonuses
are answers to a question this is not asking, which is why item 3 stayed an
if-then.

**The aggregates could not see it.** Over the same span
`gate_json.challenger_benchmarks[0].behavior.animal_spend` rises from 160 to 6,100
coins an episode and `unrecovered_animal_share` sits between 0.1 and 0.4. A growing
herd, spending well. There is no species split in that structure, so a monoculture
and a balanced herd are the same row.

## What was built

In `dqn_market_free_v2`, session `market-free-herd-v1` (a new session: the state
vector widened and `policy.load` refuses the old width).

1. **`herd.py` owns every animal role.** `herd_influenced_roles` replaces
   `goose_influenced_roles` and applies one rule to all three species: a structure
   exists for an animal that was bought, plus exactly one warm slot ahead of demand
   so the *first* purchase of any species lands the next turn rather than three
   turns later. `field_roles` is the single composer, shared by `main.py`,
   `training/clone.py` and `order_space`'s fallbacks, so no two callers can
   reconstruct a different board. Measured on a zero-animal episode: 4 pastures
   before, 1 after.

   **Superseded, and the "4 before, 1 after" was the cost rather than the win.**
   The strip that produced it is gone as of `market-free-cashherd-v1`. It deleted
   the housing that makes a purchase land, and the waste it was bought against --
   a `BUILD_` for a herd that never arrives -- was already refused by
   `donor._field_jobs`, which emits the job only at `left >= first + 2` and never
   during liquidation. Over 24 seeds in both orientations with the donor's market
   rules on both sides: 50,240 with the strip against 83,525 without, and pilkwang
   held to 86,111 instead of running to 122,515. The 52% statistic behind it was
   measured on a learner that was not buying animals, which is circular. Passes
   one, three and four -- the coop the donor cannot reach, the warm slot, the
   species pricing -- are untouched and are still what this file is about.
2. **The warm slot's species is priced over the whole herd's remaining output**, so
   MILK walks down its own `linear 1.60` curve as cows are added while EGG's `log
   0.20` barely moves. One cow scores above one goose; the second goose beats the
   second cow.
3. **Ten new state columns** (`market_state.HERD_VIABILITY_DIM`): the goose
   affordability column that block C never had, per-species unplaced holdings, and
   per-species remaining yield events and gross-per-coin. The last two are the
   quantities `ANIMAL_WASTE_WEIGHT` charges on, which the state did not carry at all.
4. **The sell ladder is depth-capped** (`order_space.MAX_SELL_SLIPPAGE = 0.15`),
   which is item 4 above. Rungs whose last unit fills more than 15% below the quote
   are shut on the v2 path -- WHEAT and EGG keep the whole ladder, WOOL and MILK do
   not. It passes the mask's proof rule by aliasing, not by measurement: the
   simulator prices each unit off the running inventory, so four small orders and
   one large one reach the same fills. The cap reprices itself off the inventory its
   own sales made, so the cumulative crash is the same arithmetic.
5. **The species split exists in the database now**: `animals_<species>_per_episode`,
   `animal_lots_<species>` and `animal_margin_<species>` in the gate's behaviour
   block. The check at the top of this file no longer needs a trace scrape.

Not built, deliberately: demonstrations, an exploration bonus on the animal columns,
and any change to the opponent pool. The payoff moved; measure before bootstrapping.
