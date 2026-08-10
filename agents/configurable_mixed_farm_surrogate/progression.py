"""Render a read-only progression and opening-behavior report from PostgreSQL."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from kaggriculture_lab.database import DatabaseTarget, connect_postgres, database_target

FAMILY_ID = "configurable_mixed_farm_surrogate"
DEFAULT_SESSION = "configurable-mixed-farm-v3-training"
DEFAULT_OPPONENT = "competitor_prvsiyan_moon_counts_melons"
DEFAULT_OUTPUT = Path("artifacts/configurable_mixed_farm_surrogate/progression_v3")


def _label(family: str) -> str:
    return {
        "competitor_prvsiyan_moon_counts_melons": "Moon",
        "configurable_crop_surrogate": "Crop surrogate",
        "receding_horizon_planner": "Receding planner",
        "builtin": "Built-ins",
    }.get(family, family.replace("competitor_", "").replace("_", " ").title())


def _frame(connection: Any, sql: str, parameters: list[Any]) -> pd.DataFrame:
    cursor = connection.execute(sql, parameters)
    columns = [column[0] for column in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def select_champion(connection: Any, session_id: str) -> str:
    row = connection.execute(
        """
        SELECT ar.revision_id
        FROM generations g
        JOIN generation_members gm USING (generation_id)
        JOIN agent_revisions ar USING (revision_id)
        JOIN rating_snapshots rs USING (generation_id, revision_id)
        WHERE g.session_id = %s AND g.status = 'COMPLETE'
          AND ar.family_id = %s AND NOT rs.is_frozen
        ORDER BY g.generation_index DESC, rs.rating - 2 * rs.deviation DESC,
                 ar.revision_id
        LIMIT 1
        """,
        [session_id, FAMILY_ID],
    ).fetchone()
    if row is None:
        raise ValueError(f"no completed mixed-farm champion in session {session_id!r}")
    return str(row[0])


def generation_progress(connection: Any, session_id: str) -> pd.DataFrame:
    frame = _frame(
        connection,
        """
        WITH paired AS (
            SELECT g.generation_index, g.status,
                   CASE WHEN a.family_id = %s THEN b.family_id ELSE a.family_id END opponent,
                   CASE WHEN a.family_id = %s THEN pr.paired_margin ELSE -pr.paired_margin END margin,
                   CASE WHEN a.family_id = %s THEN pr.outcome_a ELSE 1 - pr.outcome_a END outcome
            FROM pairing_results pr
            JOIN league_pairings lp USING (pairing_id)
            JOIN generations g USING (generation_id)
            JOIN agent_revisions a ON a.revision_id = lp.revision_a
            JOIN agent_revisions b ON b.revision_id = lp.revision_b
            WHERE g.session_id = %s
              AND (a.family_id = %s) != (b.family_id = %s)
              AND NOT pr.failed
        ), money AS (
            SELECT g.generation_index,
                   opponent.family_id AS opponent,
                   avg(ep.final_money) AS final_money
            FROM episode_players ep
            JOIN pairing_episodes pe USING (match_id)
            JOIN league_pairings lp USING (pairing_id)
            JOIN generations g USING (generation_id)
            JOIN agent_revisions policy ON policy.revision_id = ep.strategy_id
            JOIN agent_revisions opponent ON opponent.revision_id = ep.opponent_id
            WHERE g.session_id = %s AND policy.family_id = %s AND NOT ep.failed
              AND opponent.family_id != %s
            GROUP BY g.generation_index, opponent.family_id
        )
        SELECT p.generation_index, p.status, p.opponent, count(*) AS pairs,
               avg(p.margin) AS mean_margin,
               CASE WHEN count(*) > 1
                    THEN 1.96 * stddev_samp(p.margin) / sqrt(count(*)) ELSE 0 END AS margin_ci,
               avg(p.outcome) AS win_score,
               m.final_money
        FROM paired p
        LEFT JOIN money m USING (generation_index, opponent)
        GROUP BY p.generation_index, p.status, p.opponent, m.final_money
        ORDER BY p.generation_index, p.opponent
        """,
        [
            FAMILY_ID,
            FAMILY_ID,
            FAMILY_ID,
            session_id,
            FAMILY_ID,
            FAMILY_ID,
            session_id,
            FAMILY_ID,
            FAMILY_ID,
        ],
    )
    if frame.empty:
        raise ValueError(f"session {session_id!r} has no external pairing results")
    frame["opponent_label"] = frame["opponent"].map(_label)
    return frame


def surrogate_progress(connection: Any, session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    updates = _frame(
        connection,
        """
        SELECT row_number() OVER (ORDER BY mu.started_at) AS update_number,
               mu.cutoff_batch_seq, mu.status,
               (mu.metrics_json->>'candidate_weighted_error')::DOUBLE PRECISION
                   AS candidate_error,
               (mu.metrics_json->>'incumbent_weighted_error')::DOUBLE PRECISION
                   AS incumbent_error,
               (mu.metrics_json->>'moon_mae')::DOUBLE PRECISION AS moon_mae,
               mu.completed_at
        FROM model_updates mu
        JOIN training_models tm USING (model_id)
        WHERE tm.family_id = %s AND tm.model_id LIKE %s
          AND EXISTS (
              SELECT 1 FROM model_round_hooks h
              WHERE h.model_id = mu.model_id AND h.session_id = %s
          )
        ORDER BY mu.started_at
        """,
        [FAMILY_ID, f"{FAMILY_ID}.surrogate%", session_id],
    )
    conversions = _frame(
        connection,
        """
        WITH source AS (
            SELECT gm.role, count(*) AS proposed,
                   count(*) FILTER (WHERE next_elite.revision_id IS NOT NULL) AS next_elites
            FROM generation_members gm
            JOIN generations g USING (generation_id)
            LEFT JOIN generations ng
              ON ng.session_id = g.session_id
             AND ng.generation_index = g.generation_index + 1
            LEFT JOIN generation_members next_elite
              ON next_elite.generation_id = ng.generation_id
             AND next_elite.revision_id = gm.revision_id
             AND next_elite.role = 'elite'
            JOIN agent_revisions ar ON ar.revision_id = gm.revision_id
            WHERE g.session_id = %s AND g.generation_index > 0
              AND ar.family_id = %s
              AND ng.status = 'COMPLETE'
            GROUP BY gm.role
        )
        SELECT role, proposed, next_elites,
               CASE WHEN proposed > 0 THEN next_elites::DOUBLE PRECISION / proposed ELSE 0 END
                   AS elite_conversion
        FROM source ORDER BY role
        """,
        [session_id, FAMILY_ID],
    )
    return updates, conversions


def opening_records(
    connection: Any,
    revision_id: str,
    opponent_family: str,
    *,
    max_day: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = connection.execute(
        """
        SELECT ep.match_id, ep.seat, payload.replay_gzip
        FROM episode_players ep
        JOIN agent_revisions opponent ON opponent.revision_id = ep.opponent_id
        JOIN episode_payloads payload USING (match_id)
        WHERE ep.strategy_id = %s AND opponent.family_id = %s AND NOT ep.failed
        ORDER BY ep.match_id, ep.seat
        """,
        [revision_id, opponent_family],
    ).fetchall()
    if not rows:
        raise ValueError(
            f"revision {revision_id!r} has no completed episodes against {opponent_family!r}"
        )

    turns: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    for match_id, mixed_seat, raw in rows:
        replay = json.loads(gzip.decompress(bytes(raw)))
        for side, seat in (("Mixed champion", int(mixed_seat)), ("Moon", 1 - int(mixed_seat))):
            first = {
                "first_nonpass": None,
                "first_plant": None,
                "first_harvest": None,
                "first_hire": None,
                "first_land": None,
            }
            for turn, raw_step in enumerate(replay.get("steps", []) or []):
                state = raw_step[seat] if seat < len(raw_step) else {}
                observation = state.get("observation") or {}
                day = int(observation.get("day", turn // 24))
                if day > max_day:
                    break
                hour = int(observation.get("hour", turn % 24))
                action = state.get("action") or {}
                unit_actions = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
                operations = [item[0] if item else "PASS" for item in unit_actions]
                farms = observation.get("farms") or []
                farm = farms[seat] if seat < len(farms) else {}
                tiles = [tile for row in (farm.get("tiles") or []) for tile in row]
                orders = action.get("market") or []
                turns.append(
                    {
                        "match_id": match_id,
                        "side": side,
                        "turn": turn,
                        "day": day,
                        "hour": hour,
                        "money": float(farm.get("money", 0)),
                        "hands": len(farm.get("hands") or []),
                        "quadrants": len(farm.get("unlocked_quadrants") or []),
                        "plants": sum(
                            bool(isinstance(tile, dict) and tile.get("kind") == "PLANT")
                            for tile in tiles
                        ),
                        "animals": sum(
                            bool(isinstance(tile, dict) and tile.get("animal")) for tile in tiles
                        ),
                        "unit_actions": len(operations),
                        "pass_actions": sum(operation == "PASS" for operation in operations),
                        "movement_actions": sum(
                            operation in {"NORTH", "SOUTH", "EAST", "WEST"}
                            for operation in operations
                        ),
                        "productive_actions": sum(
                            operation not in {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}
                            for operation in operations
                        ),
                    }
                )
                if first["first_nonpass"] is None and any(op != "PASS" for op in operations):
                    first["first_nonpass"] = turn
                if first["first_plant"] is None and "PLANT" in operations:
                    first["first_plant"] = turn
                if first["first_harvest"] is None and "HARVEST" in operations:
                    first["first_harvest"] = turn
                if first["first_hire"] is None and any(order and order[0] == "HIRE" for order in orders):
                    first["first_hire"] = turn
                if first["first_land"] is None and any(order and order[0] == "BUY_LAND" for order in orders):
                    first["first_land"] = turn
            milestones.append({"match_id": match_id, "side": side, **first})

    return pd.DataFrame.from_records(turns), pd.DataFrame.from_records(milestones)


def opening_daily(turns: pd.DataFrame) -> pd.DataFrame:
    ordered = turns.sort_values(["match_id", "side", "day", "turn"])
    episodes = ordered.groupby(["match_id", "side", "day"], as_index=False).agg(
        end_money=("money", "last"),
        mean_hands=("hands", "mean"),
        end_plants=("plants", "last"),
        end_animals=("animals", "last"),
        end_quadrants=("quadrants", "last"),
        unit_actions=("unit_actions", "sum"),
        pass_actions=("pass_actions", "sum"),
        movement_actions=("movement_actions", "sum"),
        productive_actions=("productive_actions", "sum"),
    )
    daily = episodes.groupby(["side", "day"], as_index=False).agg(
        episodes=("match_id", "count"),
        end_money=("end_money", "mean"),
        mean_hands=("mean_hands", "mean"),
        end_plants=("end_plants", "mean"),
        end_animals=("end_animals", "mean"),
        end_quadrants=("end_quadrants", "mean"),
        unit_actions=("unit_actions", "sum"),
        pass_actions=("pass_actions", "sum"),
        movement_actions=("movement_actions", "sum"),
        productive_actions=("productive_actions", "sum"),
    )
    for category in ("pass", "movement", "productive"):
        daily[f"{category}_share"] = daily[f"{category}_actions"] / daily["unit_actions"]
    return daily


def _plot_progression(
    progress: pd.DataFrame,
    updates: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for label, group in progress.groupby("opponent_label"):
        group = group.sort_values("generation_index")
        x = group["generation_index"].to_numpy()
        margin = group["mean_margin"].to_numpy()
        ci = group["margin_ci"].fillna(0).to_numpy()
        line = axes[0, 0].plot(x, margin, marker="o", label=label)[0]
        axes[0, 0].fill_between(x, margin - ci, margin + ci, color=line.get_color(), alpha=0.12)
        axes[0, 1].plot(x, group["win_score"], marker="o", label=label)
        axes[1, 0].plot(x, group["final_money"], marker="o", label=label)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set(title="Paired margin by generation", ylabel="Mean paired margin ($)")
    axes[0, 1].axhline(0.5, color="black", linewidth=0.8)
    axes[0, 1].set(title="Win score by generation", ylabel="Win score", ylim=(-0.03, 1.03))
    axes[1, 0].set(title="Mixed-farm final money", ylabel="Mean final money ($)")
    if not updates.empty:
        axes[1, 1].plot(
            updates["cutoff_batch_seq"], updates["candidate_error"], marker="o", label="Candidate"
        )
        axes[1, 1].plot(
            updates["cutoff_batch_seq"], updates["incumbent_error"], marker="o", label="Incumbent"
        )
    axes[1, 1].set(title="Surrogate held-out error", ylabel="Weighted normalized MAE")
    for axis in axes.flat:
        axis.set_xlabel("Generation / cutoff batch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"progression.{suffix}", dpi=160)
    plt.close(fig)


def _plot_proposals(conversions: pd.DataFrame, output: Path) -> None:
    if conversions.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(conversions["role"], conversions["elite_conversion"] * 100)
    axis.set(title="Conversion into the next generation's elite", ylabel="Elite conversion (%)")
    axis.grid(axis="y", alpha=0.2)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"proposal_quality.{suffix}", dpi=160)
    plt.close(fig)


def _plot_opening(turns: pd.DataFrame, daily: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for side, group in daily.groupby("side"):
        group = group.sort_values("day")
        axes[0, 0].plot(group["day"], group["end_money"], marker="o", label=side)
        axes[0, 1].plot(group["day"], group["mean_hands"], marker="o", label=side)
        axes[1, 0].plot(group["day"], group["end_plants"], marker="o", label=f"{side} plants")
        axes[1, 0].plot(
            group["day"], group["end_animals"], marker="s", linestyle="--", label=f"{side} animals"
        )
        axes[1, 1].plot(group["day"], group["pass_share"], marker="o", label=f"{side} pass")
        axes[1, 1].plot(
            group["day"], group["productive_share"], marker="s", linestyle="--",
            label=f"{side} productive",
        )
    axes[0, 0].set(title="End-of-day money", ylabel="Money ($)")
    axes[0, 1].set(title="Average hired hands", ylabel="Hands")
    axes[1, 0].set(title="End-of-day productive assets", ylabel="Tiles / animals")
    axes[1, 1].set(title="Unit-action utilization", ylabel="Share", ylim=(0, 1))
    for axis in axes.flat:
        axis.set_xlabel("Day")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"opening.{suffix}", dpi=160)
    plt.close(fig)

    first_three = turns[turns["day"] <= 2]
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), constrained_layout=True, sharex=True)
    for axis, side in zip(axes, ("Mixed champion", "Moon")):
        group = first_three[first_three["side"] == side]
        hourly = group.groupby(["day", "hour"], as_index=False).agg(
            unit_actions=("unit_actions", "sum"), pass_actions=("pass_actions", "sum")
        )
        matrix = hourly.assign(pass_share=hourly["pass_actions"] / hourly["unit_actions"]).pivot(
            index="day", columns="hour", values="pass_share"
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="magma_r")
        axis.set(title=f"{side}: PASS share by hour", ylabel="Day", yticks=range(len(matrix.index)))
        axis.set_yticklabels(matrix.index)
    axes[-1].set_xlabel("Hour")
    fig.colorbar(image, ax=axes, label="PASS share")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"opening_heatmap.{suffix}", dpi=160)
    plt.close(fig)


def build_report(
    database: DatabaseTarget,
    session_id: str,
    output: Path,
    *,
    revision_id: str | None = None,
    opponent_family: str = DEFAULT_OPPONENT,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    connection = connect_postgres(database, initialize=False)
    try:
        champion = revision_id or select_champion(connection, session_id)
        progress = generation_progress(connection, session_id)
        updates, conversions = surrogate_progress(connection, session_id)
        turns, milestones = opening_records(connection, champion, opponent_family)
    finally:
        connection.close()
    daily = opening_daily(turns)

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))

    progress.to_csv(output / "generation_progress.csv", index=False)
    updates.to_csv(output / "surrogate_updates.csv", index=False)
    conversions.to_csv(output / "proposal_quality.csv", index=False)
    daily.to_csv(output / "opening_daily.csv", index=False)
    milestones.to_csv(output / "opening_milestones.csv", index=False)
    turns.to_csv(output / "opening_turns.csv", index=False)
    _plot_progression(progress, updates, output)
    _plot_proposals(conversions, output)
    _plot_opening(turns, daily, output)

    latest = int(progress["generation_index"].max())
    latest_status = str(progress.loc[progress["generation_index"] == latest, "status"].iloc[0])
    moon = progress[progress["opponent"] == opponent_family].sort_values("generation_index")
    early_mixed = daily[(daily["side"] == "Mixed champion") & (daily["day"].between(1, 2))]
    summary = {
        "session_id": session_id,
        "revision_id": champion,
        "opponent_family": opponent_family,
        "latest_generation": latest,
        "latest_status": latest_status,
        "moon_mean_margin_latest": None if moon.empty else float(moon.iloc[-1]["mean_margin"]),
        "opening_pass_share_days_1_2": float(
            early_mixed["pass_actions"].sum() / early_mixed["unit_actions"].sum()
        ),
        "episodes_in_opening_sample": int(turns[turns["side"] == "Mixed champion"]["match_id"].nunique()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "summary.md").write_text(
        "# Mixed-farm progression\n\n"
        f"- Session: `{session_id}`\n"
        f"- Champion revision: `{champion}`\n"
        f"- Latest generation: {latest} ({latest_status})\n"
        f"- Opening sample: {summary['episodes_in_opening_sample']} matched episodes against "
        f"{_label(opponent_family)}\n"
        f"- Champion PASS share on days 1–2: {summary['opening_pass_share_days_1_2']:.1%}\n"
        f"- Latest population margin against {_label(opponent_family)}: "
        f"{summary['moon_mean_margin_latest']:+,.0f}\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url")
    parser.add_argument("--session-id", default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision-id", default=None)
    parser.add_argument("--opponent-family", default=DEFAULT_OPPONENT)
    args = parser.parse_args(argv)
    summary = build_report(
        database_target(url=args.database_url, role="reader"),
        args.session_id,
        args.output,
        revision_id=args.revision_id,
        opponent_family=args.opponent_family,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
