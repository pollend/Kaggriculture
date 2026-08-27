import os
import sys

_TARGET = 'agents/dqn_market_free_v3d/main.py'
for _candidate in [*sys.path, os.getcwd()]:
    if _candidate and os.path.isfile(os.path.join(_candidate, _TARGET)):
        _HERE = os.path.abspath(_candidate)
        break
else:
    _HERE = os.getcwd()

while _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)
for _shadow in [
    _name
    for _name in sys.modules
    if _name == 'agents' or _name.startswith('agents.')
]:
    del sys.modules[_shadow]

from agents.dqn_market_free_v3d.main import agent as agent
