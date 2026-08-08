import math
import random

def get_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def step_toward(fx, fy, tx, ty):
    if fx > tx: return "WEST"
    if fx < tx: return "EAST"
    if fy > ty: return "NORTH"
    if fy < ty: return "SOUTH"
    return "PASS"

def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    step = obs.get("step", 0)
    money = me["money"]
    fx, fy = me["farmer"]
    
    market_orders = []
    
    # Land Expansion Logic
    unlocked = me["unlocked_quadrants"]
    if "NE" not in unlocked and money >= 1200:
        market_orders.append(["BUY_LAND"])
        money -= 1000
    elif "SW" not in unlocked and "NE" in unlocked and money >= 2300:
        market_orders.append(["BUY_LAND"])
        money -= 2000
    elif "SE" not in unlocked and "SW" in unlocked and money >= 4500:
        market_orders.append(["BUY_LAND"])
        money -= 4000

    # Seed Purchasing Strategy based on In-Game Phase
    wheat_seeds = private["seeds"].get("WHEAT", 0)
    carrot_seeds = private["seeds"].get("CARROT", 0)
    melon_seeds = private["seeds"].get("MELON", 0)
    
    if day < 5:
        if wheat_seeds < 3 and money >= 10:
            market_orders.append(["BUY_SEED", "WHEAT", 2])
            money -= 20
        elif carrot_seeds < 2 and money >= 20:
            market_orders.append(["BUY_SEED", "CARROT", 2])
            money -= 40
    elif day < 22:
        if melon_seeds < 4 and money >= 80:
            market_orders.append(["BUY_SEED", "MELON", 2])
            money -= 160
        elif wheat_seeds < 2 and money >= 10:
            market_orders.append(["BUY_SEED", "WHEAT", 2])
            money -= 20

    # Market Inventory Liquidation Strategy
    shed = private["shed"]
    for item, count in shed.items():
        if count > 0:
            market_orders.append(["SELL", item, count])

    # Enforce Market Order Queue Cap (max 10 orders per turn)
    market_orders = market_orders[:10]

    # Tile Scanning and Tactical Prioritization
    tile = me["tiles"][fy][fx]
    
    # Action 1: Clear Weeds
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return {"farmer": ["DIG"], "hands": [], "market": market_orders}
    
    # Action 2: Plant Care and Harvesting
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = day - tile["planted_day"]
        crop_type = tile["crop"]
        
        # Harvest logic
        if tile["yield_units"] > 0:
            return {"farmer": ["HARVEST"], "hands": [], "market": market_orders}
            
        # Daily watering logic
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market_orders}

    # Action 3: Planting on Empty Tiles
    if tile is None:
        if day >= 5 and private["seeds"].get("MELON", 0) > 0:
            return {"farmer": ["PLANT", "MELON"], "hands": [], "market": market_orders}
        elif private["seeds"].get("CARROT", 0) > 0:
            return {"farmer": ["PLANT", "CARROT"], "hands": [], "market": market_orders}
        elif private["seeds"].get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market_orders}

    # Movement / Target Scanning Loop
    target_x, target_y = None, None
    min_dist = 999
    for y in range(len(me["tiles"])):
        for x in range(len(me["tiles"][0])):
            t = me["tiles"][y][x]
            if t != "LOCKED":
                dist = get_distance((fx, fy), (x, y))
                if t is None and (private["seeds"].get("WHEAT", 0) > 0 or private["seeds"].get("MELON", 0) > 0):
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = x, y
                elif isinstance(t, dict) and t.get("kind") == "PLANT":
                    if not t["watered_today"] or t["yield_units"] > 0:
                        if dist < min_dist:
                            min_dist = dist
                            target_x, target_y = x, y

    if target_x is not None and target_y is not None:
        move_op = step_toward(fx, fy, target_x, target_y)
        return {"farmer": [move_op], "hands": [], "market": market_orders}

    return {"farmer": ["PASS"], "hands": [], "market": market_orders}
