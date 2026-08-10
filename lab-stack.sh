#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

container_name="kaggriculture-postgres"
volume_name="kaggriculture-postgres-data"
image_name="postgres:17-alpine"
legacy_container_name="kaggriculture-postgres-1"

container_exists() {
  docker container inspect "$container_name" >/dev/null 2>&1
}

resolve_volume() {
  if container_exists; then
    mounted_volume="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$container_name")"
    if [[ -n "$mounted_volume" ]]; then
      volume_name="$mounted_volume"
    fi
  fi
}

start_postgres() {
  if ! container_exists && docker container inspect "$legacy_container_name" >/dev/null 2>&1; then
    legacy_volume="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$legacy_container_name")"
    if [[ -n "$legacy_volume" ]]; then
      volume_name="$legacy_volume"
    fi
    docker stop "$legacy_container_name" >/dev/null 2>&1 || true
    docker rm "$legacy_container_name" >/dev/null
  fi
  docker volume create "$volume_name" >/dev/null
  if container_exists; then
    if [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" != "true" ]]; then
      docker start "$container_name" >/dev/null
    fi
  else
    docker run -d \
      --name "$container_name" \
      --restart unless-stopped \
      -e POSTGRES_DB=kaggriculture \
      -e POSTGRES_USER=kag_admin \
      -e POSTGRES_PASSWORD=kag_admin_local \
      -p 127.0.0.1:54329:5432 \
      -v "$volume_name:/var/lib/postgresql/data" \
      "$image_name" >/dev/null
  fi

  for _ in {1..60}; do
    if docker exec "$container_name" pg_isready -U kag_admin -d kaggriculture >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "PostgreSQL did not become ready within 60 seconds" >&2
  exit 1
}

resolve_volume
command_name="${1:-up}"
case "$command_name" in
  up)
    uv run python -m kaggriculture_lab.postgres_admin write-env
    start_postgres
    uv run python -m kaggriculture_lab.postgres_admin bootstrap
    exec uv run kaglab web
    ;;
  status)
    docker ps -a --filter "name=^/${container_name}$"
    ;;
  logs)
    exec docker logs -f "$container_name"
    ;;
  down)
    if container_exists; then
      if [[ "$(docker inspect --format '{{.State.Running}}' "$container_name")" == "true" ]]; then
        docker stop "$container_name" >/dev/null
      fi
    fi
    ;;
  reset)
    if [[ "${2:-}" != "--yes" ]]; then
      echo "reset deletes the PostgreSQL volume; rerun: $0 reset --yes" >&2
      exit 2
    fi
    if container_exists; then
      docker stop "$container_name" >/dev/null 2>&1 || true
      docker rm "$container_name" >/dev/null
    fi
    docker volume rm "$volume_name"
    ;;
  *)
    echo "usage: $0 {up|status|logs|down|reset --yes}" >&2
    exit 2
    ;;
esac
