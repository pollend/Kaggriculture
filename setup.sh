#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 127
fi

sync_only=false
case ${1:-} in
    "") ;;
    --sync-only)
        sync_only=true
        ;;
    *)
        echo "usage: ./setup.sh [--sync-only]" >&2
        exit 2
        ;;
esac

echo "Preparing the locked Python 3.12 environment..."
uv sync --locked --python 3.12

if [[ $sync_only == true ]]; then
    echo "Environment ready at $repo_root/.venv"
    exit 0
fi

activate="$repo_root/.venv/bin/activate"
if [[ ! -f $activate ]]; then
    echo "error: uv did not create the expected activation script: $activate" >&2
    exit 1
fi

# Activating cannot modify the shell that launched this script, so start an
# interactive child shell with the virtual environment variables inherited.
# Exiting that shell returns to the caller's original environment.
source "$activate"
interactive_shell=${SHELL:-/bin/bash}
if [[ ! -x $interactive_shell ]]; then
    interactive_shell=/bin/bash
fi

echo "Entering $VIRTUAL_ENV with $interactive_shell (exit to leave)..."
exec "$interactive_shell" -i
