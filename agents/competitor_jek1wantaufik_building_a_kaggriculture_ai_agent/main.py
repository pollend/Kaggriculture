# ==========================================================
# AUTO GENERATED SUBMISSION
# Generated : 2026-08-04T07:10:13.867679+00:00
# Source    : /kaggle/input/models/jek1wantaufik/buddy/scikitlearn/agric/2
# ==========================================================

from collections import deque
from dataclasses import dataclass
from dataclasses import dataclass, field
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
from typing import Dict, List, Tuple, Any


@dataclass(slots=True)
class GameState:
    raw: dict

    day: int
    player: int

    farm: dict
    private: dict
    market: dict

    money: int

    prices: Dict[str, int]
    shed: Dict[str, int]
    seeds: Dict[str, int]

    tiles: List[List[Any]]

    farmer: Tuple[int, int]

    hands: List

    unlocked_quadrants: List[int]

    board_size: int

    @classmethod
    def from_obs(cls, obs):
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs.get("private", {})
        market = obs.get("market", {})

        return cls(
            raw=obs,
            day=obs["day"],
            player=player,
            farm=farm,
            private=private,
            market=market,
            money=farm["money"],
            prices=market.get("prices", {}),
            shed=private.get("shed", {}),
            seeds=private.get("seeds", {}),
            tiles=farm["tiles"],
            farmer=tuple(farm["farmer"]),
            hands=farm.get("farmhands", []),
            unlocked_quadrants=farm["unlocked_quadrants"],
            board_size=len(farm["tiles"]),
        )

    @property
    def x(self):
        return self.farmer[0]

    @property
    def y(self):
        return self.farmer[1]

    @property
    def current_tile(self):
        return self.tiles[self.y][self.x]

    def price(self, item):
        return self.prices.get(item, 0)

    def inventory(self, item):
        return self.shed.get(item, 0)

    def seed_count(self, crop):
        return self.seeds.get(crop, 0)

    def has_seed(self, crop):
        return self.seed_count(crop) > 0

    def can_afford(self, amount):
        return self.money >= amount


@dataclass(slots=True)
class Tile:
    x: int
    y: int
    data: object

    @property
    def empty(self):
        return self.data is None

    @property
    def is_plant(self):
        return (
            isinstance(self.data, dict)
            and self.data.get("kind") == "PLANT"
        )

    @property
    def crop(self):
        if self.is_plant:
            return self.data["crop"]
        return None

    @property
    def watered(self):
        if self.is_plant:
            return self.data.get("watered_today", False)
        return False

    @property
    def yield_units(self):
        if self.is_plant:
            return self.data.get("yield_units", 0)
        return 0

    @property
    def planted_day(self):
        if self.is_plant:
            return self.data.get("planted_day")
        return None

    def distance(self, x, y):
        return abs(self.x - x) + abs(self.y - y)

class Board:

    def __init__(self, state):
        self.state = state
        self.tiles = state.tiles
        self.size = state.board_size

    def all_tiles(self):
        for y in range(self.size):
            for x in range(self.size):
                yield Tile(x, y, self.tiles[y][x])

    def empty_tiles(self):
        for tile in self.all_tiles():
            if tile.empty:
                yield tile

    def plants(self):
        for tile in self.all_tiles():
            if tile.is_plant:
                yield tile

    def crops(self, crop):
        for tile in self.plants():
            if tile.crop == crop:
                yield tile

    def harvestable(self):
        for tile in self.plants():
            if tile.yield_units > 0:
                yield tile

    def needs_water(self):
        for tile in self.plants():
            if not tile.watered:
                yield tile

    def nearest(self, tiles):

        fx, fy = self.state.farmer

        best = None
        best_dist = 10**9

        for tile in tiles:
            d = tile.distance(fx, fy)
            if d < best_dist:
                best_dist = d
                best = tile

        return best


@dataclass(slots=True)
class Action:
    score: float = 0.0

    farmer: list = field(default_factory=lambda: ["PASS"])
    hands: list = field(default_factory=list)
    market: list = field(default_factory=list)

    def to_dict(self):
        return {
            "farmer": self.farmer,
            "hands": self.hands,
            "market": self.market,
        }

class ActionBuilder:

    @staticmethod
    def pass_turn():
        return Action()

    @staticmethod
    def move(direction, score=0):
        return Action(
            score=score,
            farmer=[direction],
        )

    @staticmethod
    def harvest(score=0):
        return Action(
            score=score,
            farmer=["HARVEST"],
        )

    @staticmethod
    def water(score=0):
        return Action(
            score=score,
            farmer=["WATER"],
        )

    @staticmethod
    def plant(crop, score=0):
        return Action(
            score=score,
            farmer=["PLANT", crop],
        )

    @staticmethod
    def dig(score=0):
        return Action(
            score=score,
            farmer=["DIG"],
        )

    @staticmethod
    def fertilize(score=0):
        return Action(
            score=score,
            farmer=["FERTILIZE"],
        )

    @staticmethod
    def buy_seed(crop, amount, score=0):
        a = Action(score=score)
        a.market.append(["BUY_SEED", crop, amount])
        return a

    @staticmethod
    def sell(item, amount, score=0):
        a = Action(score=score)
        a.market.append(["SELL", item, amount])
        return a

    @staticmethod
    def buy_quadrant(index, score=0):
        a = Action(score=score)
        a.market.append(["BUY_QUADRANT", index])
        return a

    @staticmethod
    def hire_hand(score=0):
        a = Action(score=score)
        a.market.append(["HIRE_FARMHAND"])
        return a

    @staticmethod
    def merge(*actions):
        result = Action()

        for a in actions:

            if a.score > result.score:
                result.score = a.score

            if a.farmer != ["PASS"]:
                result.farmer = a.farmer

            result.market.extend(a.market)
            result.hands.extend(a.hands)

        return result


class Economy:

    def __init__(self, state):
        self.state = state

    # ----------------------------
    # PRICE
    # ----------------------------

    def price(self, item):
        return self.state.price(item)

    # ----------------------------
    # INVENTORY
    # ----------------------------

    def inventory(self, item):
        return self.state.inventory(item)

    def seeds(self, crop):
        return self.state.seed_count(crop)

    # ----------------------------
    # ROI
    # ----------------------------

    def crop_cost(self, crop):
        return CROPS[crop]["seed"]

    def crop_grow_days(self, crop):
        return CROPS[crop]["max_yield_day"]

    def crop_revenue(self, crop):
        return self.price(crop)

    def crop_profit(self, crop):
        return self.crop_revenue(crop) - self.crop_cost(crop)

    def crop_roi(self, crop):

        grow = max(1, self.crop_grow_days(crop))

        return self.crop_profit(crop) / grow

    # ----------------------------
    # BEST CROP
    # ----------------------------

    def best_crop(self):

        best = None
        best_roi = -1e9

        for crop in CROPS.keys():

            roi = self.crop_roi(crop)

            if roi > best_roi:
                best_roi = roi
                best = crop

        return best

    # ----------------------------
    # SELL
    # ----------------------------

    def should_sell(self, item):

        if self.inventory(item) == 0:
            return False

        return self.price(item) > self.crop_cost(item)

    # ----------------------------
    # BUY SEED
    # ----------------------------

    def should_buy_seed(self, crop):

        if self.seeds(crop) > 0:
            return False

        return self.state.can_afford(
            self.crop_cost(crop)
        )

    # ----------------------------
    # LAND
    # ----------------------------

    def should_expand(self):

        return self.state.money > 5000

    # ----------------------------
    # FARMHAND
    # ----------------------------

    def should_hire(self):

        return self.state.money > 10000


class Market:

    HISTORY = 20

    def __init__(self, state):

        self.state = state

        if not hasattr(Market, "_history"):
            Market._history = {}

        self.history = Market._history

        self._update()

    # ---------------------------------------------------------

    def _update(self):

        for item, price in self.state.prices.items():

            if item not in self.history:
                self.history[item] = deque(maxlen=self.HISTORY)

            self.history[item].append(price)

    # ---------------------------------------------------------

    def price(self, item):

        return self.state.price(item)

    # ---------------------------------------------------------

    def average(self, item):

        h = self.history.get(item)

        if not h:
            return self.price(item)

        return sum(h) / len(h)

    # ---------------------------------------------------------

    def minimum(self, item):

        h = self.history.get(item)

        if not h:
            return self.price(item)

        return min(h)

    # ---------------------------------------------------------

    def maximum(self, item):

        h = self.history.get(item)

        if not h:
            return self.price(item)

        return max(h)

    # ---------------------------------------------------------

    def trend(self, item):

        h = self.history.get(item)

        if h is None:
            return 0

        if len(h) < 2:
            return 0

        return h[-1] - h[0]

    # ---------------------------------------------------------

    def normalized_price(self, item):

        low = self.minimum(item)
        high = self.maximum(item)
        now = self.price(item)

        if high == low:
            return 0.5

        return (now - low) / (high - low)

    # ---------------------------------------------------------

    def expensive(self, item):

        return self.normalized_price(item) >= 0.80

    # ---------------------------------------------------------

    def cheap(self, item):

        return self.normalized_price(item) <= 0.20

    # ---------------------------------------------------------

    def sell_score(self, item):

        if self.state.inventory(item) == 0:
            return -999999

        score = self.price(item)

        score += self.trend(item)

        score += self.normalized_price(item) * 100

        return score

    # ---------------------------------------------------------

    def best_item_to_sell(self):

        best = None
        best_score = -1e9

        for item in self.state.shed.keys():

            s = self.sell_score(item)

            if s > best_score:

                best_score = s
                best = item

        return best


@dataclass(slots=True)
class Job:
    priority: float
    action: str
    target: tuple
    actor: str = "farmer"

class Scheduler:

    def __init__(self, state):

        self.state = state

        self.jobs = []

    # ----------------------------------------------------

    def add_job(
        self,
        action,
        x,
        y,
        priority,
        actor="farmer"
    ):

        self.jobs.append(
            Job(
                priority=priority,
                action=action,
                target=(x, y),
                actor=actor,
            )
        )

    # ----------------------------------------------------

    def sort(self):

        self.jobs.sort(
            key=lambda j: j.priority,
            reverse=True,
        )

    # ----------------------------------------------------

    def best(self):

        if not self.jobs:
            return None

        self.sort()

        return self.jobs[0]

    # ----------------------------------------------------

    def assign(self):

        self.sort()

        farmer = None

        hands = []

        used = set()

        for job in self.jobs:

            if job.target in used:
                continue

            used.add(job.target)

            if farmer is None:

                farmer = job

            else:

                hands.append(job)

        return farmer, hands

    # ----------------------------------------------------

    def clear(self):

        self.jobs.clear()


@dataclass(slots=True)
class Node:

    score: float

    action: object

    state=None

    parent=None

class Search:

    def __init__(self):

        self.nodes = []

    # --------------------------------------------------

    def clear(self):

        self.nodes.clear()

    # --------------------------------------------------

    def add(

        self,

        score,

        action,

        state=None,

        parent=None,

    ):

        self.nodes.append(

            Node(

                score=score,

                action=action,

                state=state,

                parent=parent,

            )

        )

    # --------------------------------------------------

    def empty(self):

        return len(self.nodes) == 0

    # --------------------------------------------------

    def best(self):

        if self.empty():

            return None

        return max(

            self.nodes,

            key=lambda n: n.score,

        )

    # --------------------------------------------------

    def topk(self, k=5):

        return sorted(

            self.nodes,

            key=lambda n: n.score,

            reverse=True,

        )[:k]

    # --------------------------------------------------

    def choose(self):

        node = self.best()

        if node is None:

            return None

        return node.action

    # --------------------------------------------------

    def dump(self):

        for n in self.topk():

            print(

                f"{n.score:8.2f}",

                n.action,

            )


@dataclass(slots=True)
class Candidate:
    score: float
    action: object

class Planner:

    def __init__(self, state):

        self.state = state
        self.board = Board(state)
        self.eco = Economy(state)

        self.candidates = []

    # ----------------------------------------------------

    def add(self, score, action):

        self.candidates.append(
            Candidate(score, action)
        )

    # ----------------------------------------------------

    def evaluate_market(self):

        crop = self.eco.best_crop()

        if self.eco.should_sell(crop):

            self.add(
                50,
                ActionBuilder.sell(
                    crop,
                    self.eco.inventory(crop)
                )
            )

        if self.eco.should_buy_seed(crop):

            self.add(
                40,
                ActionBuilder.buy_seed(
                    crop,
                    1
                )
            )

    # ----------------------------------------------------

    def evaluate_current_tile(self):

        tile = self.state.current_tile

        if not isinstance(tile, dict):
            return

        if tile.get("kind") != "PLANT":
            return

        if tile["yield_units"] > 0:

            value = (
                tile["yield_units"]
                * self.eco.price(tile["crop"])
            )

            self.add(
                100 + value,
                ActionBuilder.harvest()
            )

            return

        if not tile["watered_today"]:

            self.add(
                80,
                ActionBuilder.water()
            )

    # ----------------------------------------------------

    def evaluate_planting(self):

        if self.state.current_tile is not None:
            return

        crop = self.eco.best_crop()

        if not self.state.has_seed(crop):
            return

        roi = self.eco.crop_roi(crop)

        self.add(
            60 + roi,
            ActionBuilder.plant(crop)
        )

    # ----------------------------------------------------

    def evaluate_movement(self):

        target = self.board.nearest(
            self.board.harvestable()
        )

        if target:

            self.add(
                40,
                self.move_to(target)
            )

            return

        target = self.board.nearest(
            self.board.needs_water()
        )

        if target:

            self.add(
                30,
                self.move_to(target)
            )

            return

        target = self.board.nearest(
            self.board.empty_tiles()
        )

        if target:

            self.add(
                20,
                self.move_to(target)
            )

    # ----------------------------------------------------

    def move_to(self, tile):

        fx, fy = self.state.farmer

        if fx > tile.x:
            return ActionBuilder.move("WEST")

        if fx < tile.x:
            return ActionBuilder.move("EAST")

        if fy > tile.y:
            return ActionBuilder.move("NORTH")

        if fy < tile.y:
            return ActionBuilder.move("SOUTH")

        return ActionBuilder.pass_turn()

    # ----------------------------------------------------

    def choose(self):

        if not self.candidates:
            return ActionBuilder.pass_turn()

        self.candidates.sort(
            key=lambda c: c.score,
            reverse=True,
        )

        farmer = None
        market = []

        for candidate in self.candidates:

            action = candidate.action

            if farmer is None and action.farmer != ["PASS"]:
                farmer = action

            market.extend(action.market)

        if farmer is None:
            farmer = ActionBuilder.pass_turn()

        farmer.market = market

        return farmer

    # ----------------------------------------------------

    def play(self):

        self.evaluate_market()

        self.evaluate_current_tile()

        self.evaluate_planting()

        self.evaluate_movement()

        return self.choose()


def agent(obs):

    state = GameState.from_obs(obs)

    planner = Planner(state)

    return planner.play().to_dict()
