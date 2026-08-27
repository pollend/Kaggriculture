"""Per-slot member pools: N independent lineages, ranked and pruned each generation.

One learner slot used to carry one lineage, and a lineage that broke took the slot
with it. `market-free-rewardtogo-v2` ended with slot 09 at 3,000 benchmark money and
`max_abs_q` 81.8 -- a diverged Q head with nothing to fall back to -- while slot 07
reached 13,661. Reseeding existed but fired only when the learning-rate ladder decayed
past `lr_graduate_cut`, which a slot that never improves never trips.

A slot now carries `members_per_slot` members. Each is an independent lineage with its
own weights, its own trainer state and its own slice of the replay history, identified
by `lineage_id`. The generation's rounds are split between them, so each collects
`min_new_batches` batches and takes one update. At the generation boundary the slot
ranks its own checkpoints from this generation and the previous one -- `2 * N`
candidates -- keeps the top `N`, and writes them back as the next generation's members.

**Ranking is per-slot by default, and optionally across the roster.** With
`cross_slot_reseed` off, no checkpoint crosses between slots: the prune equalises
inside a slot, which fixes one lineage diverging and not one slot lagging. With it on,
a slot whose own members are all far behind may adopt a sibling slot's checkpoint --
see `foreign_candidates` and `admit_foreign` for the gate, and note the diversity cost
argued there. `market-free-rewardtogo-v2` is the case for the option: at generation 16
its ten independent slots ran from 2,086 to 18,298 rollout money with no channel
between them, so eight slots spent the whole session re-deriving what slot 07 already
had.

**The prune is elitist over two generations**, which is what makes it (mu + lambda)
rather than a plain restart: a member that got worse this generation loses to its own
parent and rolls back, and a member that broke loses to its four siblings and is
replaced by a fork of one of them.

Pure top-N is deliberate. Constraining the survivors to one per lineage collapses the
whole mechanism into "each lineage keeps the better of parent and child", under which
the members never compete and a broken lineage survives forever. The cost of allowing
duplicates is that a strong lineage can take more than one position and the slot can
converge; `lineage_cap` bounds that when it happens, and defaults to off.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Members a slot carries. `agent.toml` is the authority; this is the fallback for a
#: config that predates the key, and 1 reproduces the old one-lineage-per-slot shape
#: exactly rather than doing something new to an old run.
DEFAULT_MEMBERS = 1


@dataclass(frozen=True)
class MemberEntry:
    """One member of a slot, as persisted between generations.

    `lineage_id` rather than `position` is the identity that matters. The replay
    filter keys on it, so a member that survives a prune into a different position
    keeps its history, and a position refilled by a fork starts empty. Position is
    only the round-robin schedule and a diagnostic.
    """

    position: int
    lineage_id: str
    checkpoint_id: str | None = None
    score: float | None = None
    #: Batch sequence below which this lineage owns no history. Set to the cutoff at
    #: which the lineage was forked, so a fork does not train on its parent's data.
    batch_floor: int = 0
    generation: int = 0
    inference_path: str | None = None
    trainer_path: str | None = None
    #: The lineage this one was forked from, or None for an original member. Kept for
    #: provenance only -- nothing reads it to make a decision.
    forked_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def member_count(config: dict[str, Any]) -> int:
    """`members_per_slot`, floored at 1.

    1 is not a degenerate case: it is the old single-lineage slot, and every
    round-robin and prune path below reduces to a no-op at that value.
    """

    try:
        count = int(config.get("members_per_slot", DEFAULT_MEMBERS))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"members_per_slot must be an integer: {config.get('members_per_slot')!r}"
        ) from error
    if count < 1:
        raise ValueError(f"members_per_slot must be at least 1; got {count}")
    return count


def member_batches(config: dict[str, Any]) -> int:
    """Batches one member collects before its update fires.

    Must equal the `[[models]]` key `min_new_batches`, which is what actually makes
    the lab trigger an update -- and which the trainer cannot see, because
    `context.config` is `[models.config]` alone. Restated here rather than derived
    because there is nowhere to derive it from; `update` records the batch count it
    was actually handed so a disagreement shows up in `metrics_json` instead of
    quietly training one member on another's data.
    """

    try:
        batches = int(config.get("member_batches", 20))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"member_batches must be an integer: {config.get('member_batches')!r}"
        ) from error
    if batches < 1:
        raise ValueError(f"member_batches must be at least 1; got {batches}")
    return batches


def member_for_sequence(sequence: int, batches: int, members: int) -> int:
    """Which member owns batch `sequence`.

    The round-robin is contiguous rather than interleaved: batches
    `[0, batches)` belong to member 0, `[batches, 2 * batches)` to member 1, and the
    schedule wraps at `members`. Contiguous because a member's update consumes exactly
    the run of batches it just collected, so `context.new_batches` lands on one member
    with no filtering at the lab layer.

    The member is *derived*, never stored, because the lab's update trigger is keyed on
    `model_id` and has no member concept -- `min_new_batches` is what makes an update
    fire, so the same arithmetic has to answer for both sides.
    """

    if batches < 1:
        raise ValueError(f"batches per member must be at least 1; got {batches}")
    if members < 1:
        raise ValueError(f"members must be at least 1; got {members}")
    if sequence < 0:
        return 0
    return (sequence // batches) % members


def member_for_cutoff(cutoff: int, batches: int, members: int) -> int:
    """Which member an update at `cutoff_batch_seq` is training.

    `cutoff_batch_seq` is the 1-based sequence of the *last* batch in the window
    (`model_updates.cutoff = new_batches[-1].sequence`), while `round_index` is
    0-based, so `cutoff - 1` is the round that collected it: an update at cutoff 4
    trained sequences 1-4, collected on rounds 0-3. Expressed through
    `member_for_sequence` rather than restated, so the two sides cannot drift.

    The two sides agree only while every generation writes exactly
    `members * batches` batches for this model. A generation cut short leaves the
    cutoff counter and the per-generation round counter permanently out of phase,
    and the update then trains one member on another's batches -- which the updater
    refuses when the lineage filter leaves it nothing at all, and otherwise reports
    as `member.batches_expected != batches_received`.
    """

    return member_for_sequence(max(0, int(cutoff) - 1), batches, members)


def new_lineage_id(model_id: str, position: int, generation: int, salt: str = "") -> str:
    """A stable, collision-resistant lineage identity.

    Derived rather than random so a dry run reproduces what a live run would write,
    and content-addressed on the four things that distinguish a fork: which slot, which
    position, which generation, and -- for two forks of the same parent minted in one
    prune -- the salt.
    """

    digest = hashlib.sha256(
        "\0".join([model_id, str(position), str(generation), salt]).encode()
    )
    return digest.hexdigest()[:16]


def members_root(artifacts: Path | str) -> Path:
    return Path(artifacts) / "members"


def member_paths(artifacts: Path | str, position: int) -> tuple[Path, Path]:
    """`(inference .npz, trainer .pt)` for one member, inside this slot's artifacts."""

    root = members_root(artifacts)
    return root / f"{position}.npz", root / f"{position}.pt"


def index_path(artifacts: Path | str) -> Path:
    return members_root(artifacts) / "index.json"


def read_index(artifacts: Path | str) -> list[MemberEntry]:
    """The slot's members, or `[]` before the first prune.

    Empty is the honest answer for a cold start and for a corrupt sidecar alike: the
    caller falls back to the cold-start network either way, and a half-written index
    that raised here would take down a whole generation's collection.
    """

    path = index_path(artifacts)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    entries: list[MemberEntry] = []
    for item in payload:
        if not isinstance(item, dict) or "position" not in item or "lineage_id" not in item:
            continue
        known = {field: item.get(field) for field in MemberEntry.__dataclass_fields__}
        try:
            entries.append(MemberEntry(**known))
        except (TypeError, ValueError):
            continue
    return sorted(entries, key=lambda entry: entry.position)


def write_index(artifacts: Path | str, entries: list[MemberEntry]) -> Path:
    """Persist the member table. Written whole, never merged.

    Each slot writes only its own file, which is what makes the generation hook safe:
    `run_round_hooks` fans the hook out over every model on a thread pool, so ten
    copies run at once with no shared lock, and the only shared state a prune touches
    is the database, read-only.
    """

    path = index_path(artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.as_dict() for entry in sorted(entries, key=lambda item: item.position)]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def find_member(entries: list[MemberEntry], position: int) -> MemberEntry | None:
    for entry in entries:
        if entry.position == position:
            return entry
    return None


def _rank_key(candidate: dict[str, Any]) -> tuple:
    """Sort key: measured before unmeasured, trading before inert, then money.

    Unscored candidates sort last rather than being dropped, so a member whose rank
    arena failed is still available to fill a position -- losing a lineage to one bad
    arena run would be a harsher prune than the one being asked for. The tie-break is
    on identity, not on time, so ten concurrent hook calls agree.

    `active` is the second tier and it exists because money alone selects *for*
    silence: a member that never trades keeps its stake, so it outranks a sibling
    that traded and lost, and the roster converges on policies the release gate
    then refuses. Tiering rather than penalising keeps the money order intact
    inside each group, so when every member is inert -- the state a cold slot
    starts in -- this degrades to the old pure-money ranking instead of going
    arbitrary.

    `active is None` means no rank arena ran, which rides with the score's own
    unmeasured tier rather than counting as inert; demoting a member on absent
    evidence is the same mistake as dropping it.
    """

    score = candidate.get("score")
    scored = score is not None
    active = candidate.get("active")
    return (
        0 if scored else 1,
        0 if active is None or bool(active) else 1,
        -float(score) if scored else 0.0,
        str(candidate.get("checkpoint_id") or ""),
    )


def rank_and_prune(
    candidates: list[dict[str, Any]],
    members: int,
    *,
    model_id: str,
    generation: int,
    cutoff: int,
    lineage_cap: int | None = None,
) -> list[MemberEntry]:
    """Rank `candidates` and return the `members` that survive, one per position.

    `candidates` are this slot's checkpoints from the current and previous generation --
    `2 * members` of them once the pool is warm. Each carries `checkpoint_id`,
    `lineage_id`, `score`, `inference_path`, `trainer_path` and `batch_floor`.

    Survivors keep their own `lineage_id` and `batch_floor`, so a member that lives
    through the prune keeps its replay history. Where the same lineage takes more than
    one position, every copy after the first is a **fork**: it is given a fresh
    `lineage_id` and a `batch_floor` at the current cutoff, because a fork has no
    history of its own and training it on its parent's batches would make the two
    identical rather than merely related.

    `lineage_cap` bounds how many positions one lineage may hold. `None` is pure
    top-N -- a strong lineage can take the whole slot, which is the intended selection
    pressure and also the way this collapses. Set it to 2 if the members' scores start
    converging.
    """

    if members < 1:
        raise ValueError(f"members must be at least 1; got {members}")
    if lineage_cap is not None and lineage_cap < 1:
        raise ValueError(f"lineage_cap must be at least 1 when set; got {lineage_cap}")

    ordered = sorted(candidates, key=_rank_key)

    chosen: list[dict[str, Any]] = []
    if lineage_cap is None:
        chosen = ordered[:members]
    else:
        held: dict[str, int] = {}
        for candidate in ordered:
            if len(chosen) >= members:
                break
            lineage = str(candidate.get("lineage_id") or "")
            if held.get(lineage, 0) >= lineage_cap:
                continue
            held[lineage] = held.get(lineage, 0) + 1
            chosen.append(candidate)
        # A cap tighter than the pool can leave positions unfilled; top them up in
        # rank order rather than shipping a short roster.
        if len(chosen) < members:
            picked = {id(item) for item in chosen}
            chosen.extend(
                item for item in ordered if id(item) not in picked
            )
            chosen = chosen[:members]

    entries: list[MemberEntry] = []
    seen_lineages: set[str] = set()
    for position, candidate in enumerate(chosen):
        lineage = str(candidate.get("lineage_id") or "")
        forked_from: str | None = None
        floor = int(candidate.get("batch_floor") or 0)
        # A foreign checkpoint always forks. Its lineage id names batches written under
        # another slot's `model_id`, so keeping it would leave the replay filter
        # matching nothing here and the update would run on demonstrations alone.
        if candidate.get("foreign") or not lineage or lineage in seen_lineages:
            forked_from = lineage or None
            lineage = new_lineage_id(model_id, position, generation, salt=str(position))
            floor = int(cutoff)
        seen_lineages.add(lineage)
        entries.append(
            MemberEntry(
                position=position,
                lineage_id=lineage,
                checkpoint_id=candidate.get("checkpoint_id"),
                score=candidate.get("score"),
                batch_floor=floor,
                generation=generation,
                inference_path=candidate.get("inference_path"),
                trainer_path=candidate.get("trainer_path"),
                forked_from=forked_from,
            )
        )
    return entries


#: This slot's checkpoints at the two most recent distinct cutoffs.
#:
#: The window is expressed in `cutoff_batch_seq` rather than in generations because
#: neither `model_training_checkpoints` nor `model_updates` carries a generation id, and
#: every member advances the cutoff by exactly `min_new_batches`, so `2 * members`
#: distinct cutoffs is precisely "this generation and the previous one".
#:
#: `promoted` is deliberately not a filter. Selection decides what to *train*; the
#: release gate decides what to ship, and a checkpoint the gate rejected is still a
#: legitimate ancestor if it out-scored its siblings.
_CANDIDATE_SQL = """
WITH recent AS (
    SELECT DISTINCT mu.cutoff_batch_seq
    FROM model_updates mu
    WHERE mu.model_id = %s
    ORDER BY mu.cutoff_batch_seq DESC
    LIMIT %s
)
SELECT c.checkpoint_id,
       c.inference_artifact_path,
       c.trainer_artifact_path,
       mu.cutoff_batch_seq,
       c.metrics_json ->> 'member_lineage_id' AS lineage_id,
       CAST(c.metrics_json ->> 'member_batch_floor' AS INTEGER) AS batch_floor,
       CAST(c.metrics_json ->> 'member_rank_score' AS DOUBLE PRECISION) AS score,
       CAST(c.metrics_json ->> 'member_rank_active' AS BOOLEAN) AS active
FROM model_training_checkpoints c
JOIN model_updates mu USING (update_id)
WHERE c.model_id = %s
  AND mu.cutoff_batch_seq IN (SELECT cutoff_batch_seq FROM recent)
ORDER BY mu.cutoff_batch_seq DESC, c.checkpoint_id
"""


#: The same window over every *sibling* slot -- the other `model_id`s sharing this
#: one's `declared_model_id`, which is what a roster of named instances has in common
#: (`decl@nstep_00` ... `decl@nstep_04`).
#:
#: Ranked by score here rather than by recency, because the caller wants the roster's
#: best few and not each sibling's latest. `rn <= %s` still bounds the per-sibling
#: window first, so a slot that stopped updating cannot keep offering one stale
#: high-water checkpoint forever.
_FOREIGN_CANDIDATE_SQL = """
WITH siblings AS (
    SELECT model_id FROM training_models
    WHERE declared_model_id = (
        SELECT declared_model_id FROM training_models WHERE model_id = %s
    )
      AND model_id <> %s
),
recent AS (
    SELECT model_id,
           cutoff_batch_seq,
           row_number() OVER (
               PARTITION BY model_id ORDER BY cutoff_batch_seq DESC
           ) AS rn
    FROM (
        SELECT DISTINCT model_id, cutoff_batch_seq
        FROM model_updates
        WHERE model_id IN (SELECT model_id FROM siblings)
    ) AS distinct_cutoffs
)
SELECT c.checkpoint_id,
       c.inference_artifact_path,
       c.trainer_artifact_path,
       mu.cutoff_batch_seq,
       c.metrics_json ->> 'member_lineage_id' AS lineage_id,
       CAST(c.metrics_json ->> 'member_batch_floor' AS INTEGER) AS batch_floor,
       CAST(c.metrics_json ->> 'member_rank_score' AS DOUBLE PRECISION) AS score,
       CAST(c.metrics_json ->> 'member_rank_active' AS BOOLEAN) AS active,
       c.model_id
FROM model_training_checkpoints c
JOIN model_updates mu USING (update_id)
JOIN recent r ON r.model_id = mu.model_id AND r.cutoff_batch_seq = mu.cutoff_batch_seq
WHERE r.rn <= %s
  AND CAST(c.metrics_json ->> 'member_rank_score' AS DOUBLE PRECISION) IS NOT NULL
ORDER BY score DESC, c.checkpoint_id
LIMIT %s
"""


def _row_to_candidate(row) -> dict[str, Any] | None:
    """One SQL row to a candidate dict, or None if its artifacts are gone.

    A row whose `.npz` or `.pt` has gone missing is dropped rather than raised on: the
    artifact roots are cleaned by hand between experiments, and a prune that died on a
    stale row would take the generation with it.
    """

    inference, trainer = str(row[1] or ""), str(row[2] or "")
    if not inference or not Path(inference).is_file():
        return None
    if not trainer or not Path(trainer).is_file():
        return None
    return {
        "checkpoint_id": str(row[0]),
        "inference_path": inference,
        "trainer_path": trainer,
        "cutoff_batch_seq": int(row[3] or 0),
        "lineage_id": str(row[4]) if row[4] else "",
        "batch_floor": int(row[5] or 0),
        "score": float(row[6]) if row[6] is not None else None,
        # Absent on checkpoints written before the activity tier existed, and None is
        # the tier that neither promotes nor demotes them.
        "active": bool(row[7]) if row[7] is not None else None,
    }


def foreign_candidates(reader, model_id: str, members: int, limit: int) -> list[dict[str, Any]]:
    """The roster's best checkpoints from *other* slots, tagged `foreign`.

    `batch_floor` is deliberately absent: a foreign checkpoint's own floor belongs to a
    batch stream this `model_id` never wrote, and `rank_and_prune` forces every foreign
    survivor through the fork path, which sets the floor to the current cutoff. Adopting
    one without that would leave the member's replay filter matching nothing at all --
    the lineage id names batches under another slot's `model_id` -- and the update would
    then train on demonstrations alone.
    """

    if limit < 1:
        return []
    rows = reader.fetchall(
        _FOREIGN_CANDIDATE_SQL, [model_id, model_id, 2 * members, limit]
    ) or []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = _row_to_candidate(row)
        if candidate is None:
            continue
        candidate["foreign"] = True
        candidate["source_model_id"] = str(row[8])
        candidates.append(candidate)
    return candidates


def admit_foreign(
    own: list[dict[str, Any]],
    foreign: list[dict[str, Any]],
    *,
    min_score: float,
    margin: float,
    max_adopt: int,
) -> list[dict[str, Any]]:
    """Which foreign checkpoints this slot is allowed to consider, best first.

    Three gates, and all of them exist to stop the roster collapsing onto one policy:

    * **`active`** must be measured true. Money alone selects *for* silence: a policy
      that never trades keeps its whole stake, so it scores the stake exactly and
      outranks every sibling that traded and lost. `_rank_key` tiers on this for the
      same reason. Importing is a bigger step than keeping what you have, so an
      unmeasured checkpoint (`active is None`) is refused too -- a slot must not adopt
      on absent evidence. `market-free-top25-v1` is why this is a gate and not a
      preference: an inert checkpoint scoring exactly 3,000 was forked into position 0
      of all five slots at the g0 prune, and every one then collected at 0.11 orders
      per decision against its sibling's 0.53.
    * **`min_score`** is an absolute floor, compared *strictly*. Without it two dead
      slots swap corpses -- every score near zero compares favourably with some other
      score near zero. It must sit above the stake and exclude it: the stake is exactly
      what an idle policy scores, so `>=` at the stake admits the one candidate this
      exists to refuse.
    * **`margin`** is the fractional lead a foreign checkpoint must hold over this
      slot's *own* best before it is worth importing. A slot that is doing fine keeps
      what it has; only a slot that is far behind reaches across.

    `max_adopt` bounds how many positions may be taken. The caller should keep it below
    `members` when it wants a slot to retain one of its own lineages, because the
    independent-replicate property is the thing the roster is actually buying: on
    `market-free-rewardtogo-v2` eight of ten slots finished under 7,000 and two carried
    the run, and a rule that lets every slot converge on this generation's leader spends
    that hedge. Adopted copies fork and diverge again through their own noise and their
    own batch stream, but they start from one point.
    """

    if max_adopt < 1 or not foreign:
        return []
    own_best = max(
        (float(item["score"]) for item in own if item.get("score") is not None),
        default=None,
    )
    bar = min_score if own_best is None else max(min_score, own_best * (1.0 + margin))
    admitted = [
        item
        for item in foreign
        if item.get("score") is not None
        and item.get("active") is True
        and float(item["score"]) > bar
    ]
    admitted.sort(key=lambda item: -float(item["score"]))
    return admitted[:max_adopt]


def roster_candidates(reader, model_id: str, members: int) -> list[dict[str, Any]]:
    """Prune candidates for one slot, artifact-existence filtered.

    A row whose `.npz` or `.pt` has gone missing is dropped rather than raised on: the
    artifact roots are cleaned by hand between experiments, and a prune that died on a
    stale row would take the generation with it.
    """

    rows = reader.fetchall(_CANDIDATE_SQL, [model_id, 2 * members, model_id]) or []
    return [
        candidate
        for candidate in (_row_to_candidate(row) for row in rows)
        if candidate is not None
    ]
