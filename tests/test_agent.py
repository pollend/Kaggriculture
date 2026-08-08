from copy import deepcopy

from agents.configurable_crop_surrogate.agent import make_agent
from agents.configurable_crop_surrogate.strategy import Strategy


def observation(*, hour=0, step=0, hands=None, tile=None, seeds=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    for y in range(10):
        for x in range(10):
            if x >= 5 or y >= 5:
                tiles[y][x] = "LOCKED"
    tiles[4][4] = tile
    hands = hands or []
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": hour,
        "farms": [
            {
                "money": 3000,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": hands,
                "hires_today": len(hands),
                "unlocked_quadrants": ["NW"],
            },
            {
                "money": 3000,
                "tiles": deepcopy(tiles),
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": {crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
            "seeds": seeds or {crop: 0 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
        "market": {
            "prices": {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250}
        },
        "town": {"unlocked_shops": []},
    }


def test_agent_returns_one_action_per_hand():
    obs = observation(hands=[[3, 4], [4, 3]])
    result = make_agent(Strategy())(obs)
    assert set(result) == {"farmer", "hands", "market"}
    assert len(result["hands"]) == 2
    assert len(result["market"]) <= 10


def test_emergency_watering_precedes_harvest():
    plant = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 4,
    }
    result = make_agent(Strategy())(observation(step=72, tile=plant))
    assert result["farmer"] == ["WATER"]


def test_agent_does_not_plant_on_last_hour():
    obs = observation(hour=23, step=23, seeds={"WHEAT": 1})
    result = make_agent(Strategy())(obs)
    assert result["farmer"][0] != "PLANT"


def test_endgame_sells_and_does_not_buy():
    obs = observation(step=700)
    obs["private"]["shed"]["WHEAT"] = 4
    result = make_agent(Strategy())(obs)
    assert ["SELL", "WHEAT", 4] in result["market"]
    assert not any(order[0].startswith("BUY") for order in result["market"])


def test_episode_state_resets_when_a_new_episode_starts():
    agent = make_agent(Strategy())
    agent(observation(step=1))
    agent(observation(step=2))
    assert sum(agent.state.action_counts.values()) == 2
    agent(observation(step=0))
    assert agent.state.last_step == 0
    assert sum(agent.state.action_counts.values()) == 1
