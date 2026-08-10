from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from agents.mixed_farm_v5.progression import opening_daily, opening_records
from kaggriculture_lab.database import connect_postgres, database_target


def _replay() -> dict:
    steps = []
    for turn in range(48):
        states = []
        for seat in range(2):
            farm = {
                "money": 3000 + turn * (seat + 1),
                "hands": [[3, 4]] if turn >= 1 else [],
                "tiles": [[{"kind": "PLANT", "crop": "WHEAT"}] if turn >= 2 else [None]],
                "unlocked_quadrants": ["NW"],
            }
            states.append(
                {
                    "observation": {
                        "day": turn // 24,
                        "hour": turn % 24,
                        "farms": [farm, farm],
                    },
                    "action": {
                        "farmer": ["PLANT", "WHEAT"] if turn == 2 else ["PASS"],
                        "hands": [["PASS"]] if turn >= 1 else [],
                        "market": [["HIRE"]] if turn == 1 else [],
                    },
                }
            )
        steps.append(states)
    return {"steps": steps}


def test_opening_records_decompresses_replay_and_aggregates_days():
    try:
        connection = connect_postgres(database_target("experiments"), initialize=False)
    except Exception as error:
        pytest.skip(f"local PostgreSQL lab stack is unavailable: {error}")
    connection.execute("BEGIN")
    connection.execute("SET LOCAL search_path TO pg_temp")
    connection.execute(
        "CREATE TABLE agent_revisions (revision_id VARCHAR, family_id VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE episode_players (match_id VARCHAR, seat INTEGER, strategy_id VARCHAR, "
        "opponent_id VARCHAR, failed BOOLEAN)"
    )
    connection.execute("CREATE TABLE episode_payloads (match_id VARCHAR, replay_gzip BYTEA)")
    connection.execute(
        "INSERT INTO agent_revisions VALUES ('mixed', 'mixed_farm_v5'), "
        "('moon', 'competitor_prvsiyan_moon_counts_melons')"
    )
    connection.execute("INSERT INTO episode_players VALUES ('m1', 0, 'mixed', 'moon', false)")
    connection.execute(
        "INSERT INTO episode_payloads VALUES ('m1', %s)",
        [gzip.compress(json.dumps(_replay()).encode())],
    )

    turns, milestones = opening_records(
        connection, "mixed", "competitor_prvsiyan_moon_counts_melons", max_day=1
    )
    daily = opening_daily(turns)

    assert set(turns["side"]) == {"Mixed champion", "Moon"}
    assert len(daily) == 4
    assert milestones.loc[milestones["side"] == "Mixed champion", "first_plant"].item() == 2
    assert daily["pass_share"].between(0, 1).all()
    connection.execute("ROLLBACK")
    connection.close()


def test_opening_daily_uses_end_state_and_action_weighting():
    turns = pd.DataFrame(
        [
            {"match_id": "a", "side": "Mixed champion", "day": 0, "turn": 0,
             "money": 10, "hands": 0, "plants": 0, "animals": 0, "quadrants": 1,
             "unit_actions": 1, "pass_actions": 1, "movement_actions": 0,
             "productive_actions": 0},
            {"match_id": "a", "side": "Mixed champion", "day": 0, "turn": 1,
             "money": 20, "hands": 1, "plants": 2, "animals": 1, "quadrants": 1,
             "unit_actions": 2, "pass_actions": 0, "movement_actions": 1,
             "productive_actions": 1},
        ]
    )
    daily = opening_daily(turns).iloc[0]
    assert daily["end_money"] == 20
    assert daily["end_plants"] == 2
    assert daily["pass_share"] == 1 / 3
