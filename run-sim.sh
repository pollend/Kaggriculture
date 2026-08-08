#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 127
fi

echo "Preparing the locked Python 3.12 environment..."
uv sync --python 3.12

if [[ ${1:-} == "setup" ]]; then
    echo "Environment ready at $repo_root/.venv"
    exit 0
fi

if [[ $# -eq 0 ]]; then
    set -- match agents/configurable_crop_surrogate/main.py starter --seeds 10
elif [[ $1 == "competitors" ]]; then
    shift
    competitor_agents=(agents/competitor_*/main.py)
    if [[ ! -e ${competitor_agents[0]} ]]; then
        echo "error: no competitor agents found under agents/competitor_*" >&2
        exit 1
    fi
    set -- league \
        agents/configurable_crop_surrogate/main.py \
        "${competitor_agents[@]}" \
        starter \
        --seeds 10 \
        "$@"
fi

echo "Running: python -m sim.arena $*"
exec uv run --no-sync python -m sim.arena "$@"
