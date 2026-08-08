"""Code-aware immutable identities for agents entered into training leagues."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

DEFAULT_AGENT_NAME = "default"
SAFE_AGENT_NAME = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class TrainingAgentSpec:
    """One independently persisted training instance of an agent family."""

    type: str
    name: str = DEFAULT_AGENT_NAME
    primary: bool = False
    train: bool = True
    seed: int | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingAgentSpec:
        return cls(
            type=str(values["type"]),
            name=str(values.get("name", DEFAULT_AGENT_NAME)),
            primary=bool(values.get("primary", False)),
            train=bool(values.get("train", True)),
            seed=(None if values.get("seed") is None else int(values["seed"])),
        )

    def validate(self) -> None:
        if not SAFE_AGENT_NAME.fullmatch(self.type):
            raise ValueError(f"invalid training agent type: {self.type!r}")
        if not SAFE_AGENT_NAME.fullmatch(self.name):
            raise ValueError(f"invalid training agent name: {self.name!r}")
        if self.seed is not None and self.seed < 0:
            raise ValueError("training agent seed must be non-negative")


def effective_model_id(declared_model_id: str, agent_name: str) -> str:
    """Namespace a manifest model ID while preserving legacy default IDs."""

    if agent_name == DEFAULT_AGENT_NAME:
        return declared_model_id
    return f"{declared_model_id}@{agent_name}"


def training_instance_seed(agent_type: str, agent_name: str) -> int:
    """Return a reproducible, distinct seed for a named training instance."""

    digest = hashlib.sha256(f"{agent_type}\0{agent_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def training_artifact_root(
    artifacts: str | Path, config: dict[str, Any]
) -> Path:
    """Scope model artifacts to a validated type/name directory when present."""

    agent_type = config.get("agent_type")
    agent_name = config.get("name")
    if agent_type is None or agent_name is None:
        return Path(artifacts)
    if not SAFE_AGENT_NAME.fullmatch(str(agent_type)):
        raise ValueError(f"invalid configured training agent type: {agent_type!r}")
    if not SAFE_AGENT_NAME.fullmatch(str(agent_name)):
        raise ValueError(f"invalid configured training agent name: {agent_name!r}")
    return Path(artifacts) / str(agent_type) / str(agent_name)


@dataclass(frozen=True, slots=True)
class TrainingModelManifest:
    """One trainable model and the single object the lab enters it through.

    `trainer` names an object exposing `on_batch` (required) and optionally
    `on_round`. It replaced the earlier `updater` + `collector` pair, which split
    one agent's training across two seams with different privileges.
    """

    model_id: str
    trainer: str
    min_new_batches: int
    config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingModelManifest:
        retired = sorted({"updater", "collector"} & set(values))
        if retired:
            raise ValueError(
                f"[[models]] {', '.join(retired)} was replaced by a single "
                "trainer = \"agents.<name>.train:trainer\""
            )
        minimum = int(values.get("min_new_batches", 1))
        if minimum < 1:
            raise ValueError("model min_new_batches must be positive")
        return cls(
            model_id=str(values["model_id"]),
            trainer=str(values["trainer"]),
            min_new_batches=minimum,
            config=dict(values.get("config", {})),
        )


@dataclass(frozen=True, slots=True)
class AgentManifest:
    family_id: str
    name: str
    entry_point: str
    parameter_schema_version: int
    code_paths: tuple[str, ...]
    training_adapter: str | None = None
    episode_evaluator: str | None = None
    models: tuple[TrainingModelManifest, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> AgentManifest:
        with Path(path).open("rb") as stream:
            values = tomllib.load(stream)
        return cls(
            family_id=str(values["family_id"]),
            name=str(values["name"]),
            entry_point=str(values["entry_point"]),
            parameter_schema_version=int(values.get("parameter_schema_version", 1)),
            code_paths=tuple(str(item) for item in values.get("code_paths", ())),
            training_adapter=values.get("training_adapter"),
            episode_evaluator=values.get("episode_evaluator"),
            models=tuple(
                TrainingModelManifest.from_dict(item)
                for item in values.get("models", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentRevision:
    revision_id: str
    family_id: str
    strategy_id: str | None
    entry_point: str
    agent_spec: str
    code_fingerprint: str
    parameter_schema_version: int
    params_json: str | None
    checkpoint_id: str | None = None
    checkpoint_digest: str | None = None
    source: str = "generated"
    episode_evaluator: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "family_id": self.family_id,
            "strategy_id": self.strategy_id,
            "entry_point": self.entry_point,
            "agent_spec": self.agent_spec,
            "code_fingerprint": self.code_fingerprint,
            "parameter_schema_version": self.parameter_schema_version,
            "params_json": self.params_json,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_digest": self.checkpoint_digest,
            "source": self.source,
            "episode_evaluator": self.episode_evaluator,
        }


def code_fingerprint(manifest: AgentManifest, workspace: str | Path = ".") -> str:
    """Hash every declared behavior-defining file in stable path order."""

    root = Path(workspace).resolve()
    files: set[Path] = set()
    for declared in manifest.code_paths:
        candidate = (root / declared).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"code path escapes workspace: {declared}")
        if candidate.is_dir():
            files.update(
                item
                for item in candidate.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
            )
        elif candidate.is_file():
            files.add(candidate)
        else:
            raise FileNotFoundError(f"manifest code path does not exist: {declared}")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_revision(
    manifest: AgentManifest,
    strategy: Any | None = None,  # family params; only canonical_json()/strategy_id are used
    *,
    workspace: str | Path = ".",
    episode_steps: int = 720,
    checkpoint_id: str | None = None,
    checkpoint_digest: str | None = None,
    source: str = "generated",
) -> AgentRevision:
    fingerprint = code_fingerprint(manifest, workspace)
    params = strategy.canonical_json() if strategy else None
    strategy_id = strategy.strategy_id if strategy else None
    identity_payload = json.dumps(
        {
            "family_id": manifest.family_id,
            "entry_point": manifest.entry_point,
            "code_fingerprint": fingerprint,
            "parameter_schema_version": manifest.parameter_schema_version,
            "params_json": params,
            "checkpoint_digest": checkpoint_digest,
            "episode_evaluator": manifest.episode_evaluator,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    revision_id = hashlib.sha256(identity_payload.encode()).hexdigest()[:24]
    if strategy is None:
        spec = manifest.entry_point
    else:
        query = urlencode({**strategy.to_dict(), "episode_steps": episode_steps})
        spec = f"{manifest.entry_point}?{query}"
    return AgentRevision(
        revision_id=revision_id,
        family_id=manifest.family_id,
        strategy_id=strategy_id,
        entry_point=manifest.entry_point,
        agent_spec=spec,
        code_fingerprint=fingerprint,
        parameter_schema_version=manifest.parameter_schema_version,
        params_json=params,
        checkpoint_id=checkpoint_id,
        checkpoint_digest=checkpoint_digest,
        source=source,
        episode_evaluator=manifest.episode_evaluator,
    )


def builtin_revision(name: str, spec: str | None = None) -> AgentRevision:
    fingerprint = hashlib.sha256(f"builtin:{name}".encode()).hexdigest()
    return AgentRevision(
        revision_id=f"builtin:{name}",
        family_id="builtin",
        strategy_id=None,
        entry_point=spec or name,
        agent_spec=spec or name,
        code_fingerprint=fingerprint,
        parameter_schema_version=1,
        params_json=None,
        source="builtin",
        episode_evaluator=None,
    )
