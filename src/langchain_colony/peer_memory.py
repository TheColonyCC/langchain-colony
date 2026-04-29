"""Persistent peer-summary memory for langchain-colony agents.

Each peer the agent has interacted with gets a small ``PeerSummary``
record. Updates happen in two phases: a cheap structured update on
every observation (counters, recent-positions ring, relationship state
machine), and an optional LLM-distillation pass every K-th interaction
that refreshes ``style_notes``.

This module is the Python sibling of the ElizaOS plugin's
``peer-memory.ts`` (v0.31.0). Same shape, same observation kinds, same
mechanical relationship state machine — kept symmetric so reasoning
about "what does the agent know about this peer" is consistent across
both stacks.

Library shape: this module ships *primitives*. The consumer wires them
into their dispatch path. ``JSONFilePeerMemoryStore`` is the default
implementation; users with different persistence needs implement the
``PeerMemoryStore`` Protocol.

Privacy: stored summaries are derived metadata — the agent's private
notes about how peers behave, not republished content. The
``format_for_prompt`` block instructs the model not to cite the notes
verbatim, and ``recent_positions`` entries are short paraphrases
truncated at 200 chars.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

PEER_MEMORY_DEFAULT_DIR = Path.home() / ".langchain-colony"
RECENT_POSITIONS_RING = 3
POSITION_MAX_CHARS = 200
STYLE_NOTES_MAX_CHARS = 500
TOP_TOPICS_FOR_PROMPT = 3

PeerObservationKind = Literal[
    "engagement-comment",
    "watched-comment",
    "dm-received",
    "dm-reply-sent",
    "comment-on-self",
    "auto-upvote",
    "auto-downvote",
    "manual-vote",
]

Relationship = Literal["neutral", "agreed", "disagreed", "mixed"]


@dataclass
class VoteHistory:
    up: int = 0
    down: int = 0


@dataclass
class PeerSummary:
    """Per-peer record. Mirrors the v0.31.0 TypeScript ``PeerSummary``."""

    username: str
    first_seen: float
    last_seen: float
    interaction_count: int = 0
    topics: dict[str, int] = field(default_factory=dict)
    vote_history: VoteHistory = field(default_factory=VoteHistory)
    style_notes: str = ""
    recent_positions: list[str] = field(default_factory=list)
    relationship: Relationship = "neutral"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vote_history"] = asdict(self.vote_history)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PeerSummary:
        vh = data.get("vote_history") or {}
        return cls(
            username=data["username"],
            first_seen=float(data.get("first_seen", 0)),
            last_seen=float(data.get("last_seen", 0)),
            interaction_count=int(data.get("interaction_count", 0)),
            topics=dict(data.get("topics") or {}),
            vote_history=VoteHistory(
                up=int(vh.get("up", 0)),
                down=int(vh.get("down", 0)),
            ),
            style_notes=str(data.get("style_notes", "")),
            recent_positions=list(data.get("recent_positions") or []),
            relationship=data.get("relationship", "neutral"),
        )


@dataclass
class PeerObservation:
    """A single interaction event to fold into the peer's summary."""

    kind: PeerObservationKind
    topics: list[str] | None = None
    position: str | None = None


def new_summary(username: str, now: float) -> PeerSummary:
    """Fresh summary at default state for a previously-unseen peer."""
    return PeerSummary(
        username=username,
        first_seen=now,
        last_seen=now,
    )


def compute_relationship(vh: VoteHistory, interaction_count: int) -> Relationship:
    """Mechanical relationship state machine.

    Not LLM-derived — the structured signals from auto-vote outcomes
    are enough. ``neutral`` until ≥3 interactions; then ``agreed`` /
    ``disagreed`` on a ≥2-vote net, ``mixed`` on at least one of each,
    ``neutral`` otherwise.
    """
    if interaction_count < 3:
        return "neutral"
    delta = vh.up - vh.down
    if delta >= 2:
        return "agreed"
    if delta <= -2:
        return "disagreed"
    if vh.up >= 1 and vh.down >= 1:
        return "mixed"
    return "neutral"


def apply_observation(
    existing: PeerSummary,
    obs: PeerObservation,
    now: float,
) -> PeerSummary:
    """Pure structured-update phase. Returns a new ``PeerSummary``.

    Increments counters, updates the ``recent_positions`` ring (max 3,
    most-recent-first, dedup'd), tallies ``topics`` and
    ``vote_history`` from the observation kind, and recomputes
    ``relationship`` from the new totals. Does NOT call the LLM —
    distillation is a separate phase the caller orchestrates.
    """
    # Defensive copy so the caller's input isn't mutated.
    next_summary = PeerSummary(
        username=existing.username,
        first_seen=existing.first_seen,
        last_seen=now,
        interaction_count=existing.interaction_count + 1,
        topics=dict(existing.topics),
        vote_history=VoteHistory(
            up=existing.vote_history.up,
            down=existing.vote_history.down,
        ),
        style_notes=existing.style_notes,
        recent_positions=list(existing.recent_positions),
        relationship=existing.relationship,
    )

    if obs.topics:
        for raw in obs.topics:
            key = str(raw).lower().strip()
            if not key:
                continue
            next_summary.topics[key] = next_summary.topics.get(key, 0) + 1

    if obs.position:
        truncated = obs.position[:POSITION_MAX_CHARS].strip()
        if truncated:
            # Most-recent-first; dedup the same position to the front.
            existing_positions = [p for p in next_summary.recent_positions if p != truncated]
            next_summary.recent_positions = [truncated, *existing_positions][:RECENT_POSITIONS_RING]

    if obs.kind in ("auto-upvote", "manual-vote"):
        next_summary.vote_history.up += 1
    elif obs.kind == "auto-downvote":
        next_summary.vote_history.down += 1

    next_summary.relationship = compute_relationship(
        next_summary.vote_history,
        next_summary.interaction_count,
    )
    return next_summary


def prune_stale(
    peer_map: dict[str, PeerSummary],
    ttl_seconds: float,
    now: float,
) -> dict[str, PeerSummary]:
    """Drop peers whose ``last_seen`` is older than ``ttl_seconds``.

    Pure; the caller decides when to invoke (typically before each
    cache write).
    """
    if ttl_seconds <= 0:
        return peer_map
    cutoff = now - ttl_seconds
    return {k: v for k, v in peer_map.items() if v.last_seen >= cutoff}


def cap_by_last_seen(
    peer_map: dict[str, PeerSummary],
    max_peers: int,
) -> dict[str, PeerSummary]:
    """LRU-by-``last_seen`` cap.

    When the map exceeds ``max_peers``, drop the oldest. Pure — caller
    decides cadence.
    """
    if max_peers <= 0:
        return peer_map
    if len(peer_map) <= max_peers:
        return peer_map
    sorted_pairs = sorted(
        peer_map.items(),
        key=lambda kv: kv[1].last_seen,
        reverse=True,
    )
    return dict(sorted_pairs[:max_peers])


def format_for_prompt(summary: PeerSummary, now: float) -> str:
    """Render a peer summary into the private-context block.

    Returned string is prepended to engagement / DM-reply prompts.
    Returns ``""`` when the summary is too thin to be useful (zero
    interactions), so the caller can suppress the block cleanly.
    """
    if summary.interaction_count <= 0:
        return ""

    age_seconds = max(86400.0, now - summary.last_seen)
    age_days = max(1, round(age_seconds / 86400))

    top_topics = [
        k for k, _ in sorted(summary.topics.items(), key=lambda kv: kv[1], reverse=True)[:TOP_TOPICS_FOR_PROMPT]
    ]

    interaction_word = "interaction" if summary.interaction_count == 1 else "interactions"
    day_word = "day" if age_days == 1 else "days"

    lines = [
        f"Context on @{summary.username} (private — do NOT cite verbatim or reference these notes explicitly):",
        f"- Last interacted: {age_days} {day_word} ago, {summary.interaction_count} prior {interaction_word}",
    ]
    if top_topics:
        lines.append(f"- Topics they care about: {', '.join(top_topics)}")
    if summary.style_notes:
        lines.append(f"- Notes: {summary.style_notes}")
    if summary.recent_positions:
        lines.append(f"- Recent positions: {' | '.join(summary.recent_positions)}")
    lines.append(f"- Relationship: {summary.relationship}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Storage Protocol
# ──────────────────────────────────────────────────────────────────────


class PeerMemoryStore(Protocol):
    """Persistence interface for the peer-memory map.

    The default implementation is ``JSONFilePeerMemoryStore``. Users
    with different persistence needs (SQLite, Redis, S3, etc.) supply
    their own.
    """

    def get_map(self) -> dict[str, PeerSummary]: ...

    def save_map(self, peer_map: dict[str, PeerSummary]) -> None: ...

    def get_summary(self, username: str) -> PeerSummary | None: ...

    def record_observation(
        self,
        peer_username: str | None,
        observation: PeerObservation,
        *,
        distill_every: int = 5,
        distillation_callback: Callable[[PeerSummary], str | None] | None = None,
        max_peers: int = 200,
        ttl_seconds: float = 90 * 86400,
        self_username: str | None = None,
        now: float | None = None,
    ) -> PeerSummary | None: ...

    def format_for_prompt(self, username: str | None, now: float | None = None) -> str: ...


# ──────────────────────────────────────────────────────────────────────
# Default implementation: JSON file backed
# ──────────────────────────────────────────────────────────────────────


class JSONFilePeerMemoryStore:
    """Single-file JSON-backed ``PeerMemoryStore``.

    Stores the entire ``Record<peer_username, PeerSummary>`` map as one
    JSON file. Loaded on init; rewritten on every ``save_map``. Single-
    threaded, no locking — fine for the single-agent-per-host pattern
    that langchain-colony users follow today.

    A corrupted file is logged-and-treated-as-empty rather than raised
    — a transient I/O glitch shouldn't crash an autonomy loop.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._map: dict[str, PeerSummary] | None = None

    # ── load / save ──────────────────────────────────────────────────

    def _load(self) -> dict[str, PeerSummary]:
        if self._map is not None:
            return self._map
        if not self.path.exists():
            self._map = {}
            return self._map
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "PEER_MEMORY: failed to load %s (%s) — treating as empty",
                self.path,
                exc,
            )
            self._map = {}
            return self._map
        if not isinstance(data, dict):
            logger.warning(
                "PEER_MEMORY: %s does not contain a JSON object — treating as empty",
                self.path,
            )
            self._map = {}
            return self._map
        out: dict[str, PeerSummary] = {}
        for username, entry in data.items():
            if not isinstance(entry, dict):
                continue
            try:
                out[username] = PeerSummary.from_dict(entry)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "PEER_MEMORY: skipping malformed entry for @%s in %s: %s",
                    username,
                    self.path,
                    exc,
                )
        self._map = out
        return out

    def _save(self, peer_map: dict[str, PeerSummary]) -> None:
        self._map = peer_map
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {k: v.to_dict() for k, v in peer_map.items()}
        # Atomic write so a crash mid-write doesn't leave a half-file.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    # ── PeerMemoryStore Protocol ─────────────────────────────────────

    def get_map(self) -> dict[str, PeerSummary]:
        return dict(self._load())

    def save_map(self, peer_map: dict[str, PeerSummary]) -> None:
        self._save(peer_map)

    def get_summary(self, username: str) -> PeerSummary | None:
        if not username:
            return None
        return self._load().get(username)

    def record_observation(
        self,
        peer_username: str | None,
        observation: PeerObservation,
        *,
        distill_every: int = 5,
        distillation_callback: Callable[[PeerSummary], str | None] | None = None,
        max_peers: int = 200,
        ttl_seconds: float = 90 * 86400,
        self_username: str | None = None,
        now: float | None = None,
    ) -> PeerSummary | None:
        """Record a single observation. Two-phase: cheap structured
        update always runs; optional LLM distillation runs only every
        K-th interaction.

        Returns the updated summary, or ``None`` if recording was a
        no-op (no username, or peer is self). Non-throwing — internal
        errors are logged rather than raised.
        """
        if not peer_username:
            return None
        if self_username and peer_username == self_username:
            return None
        ts = now if now is not None else time.time()
        try:
            peer_map = self._load()
            existing = peer_map.get(peer_username) or new_summary(peer_username, ts)
            updated = apply_observation(existing, observation, ts)

            distill_clamped = max(1, min(50, int(distill_every))) if distill_every else 5
            if (
                updated.interaction_count > 0
                and updated.interaction_count % distill_clamped == 0
                and distillation_callback is not None
            ):
                try:
                    distilled = distillation_callback(updated)
                except Exception as exc:
                    logger.warning(
                        "PEER_MEMORY: distillation callback raised for @%s: %s",
                        peer_username,
                        exc,
                    )
                    distilled = None
                if distilled and distilled.strip():
                    updated.style_notes = distilled.strip()[:STYLE_NOTES_MAX_CHARS]

            peer_map[peer_username] = updated
            pruned = cap_by_last_seen(prune_stale(peer_map, ttl_seconds, ts), max_peers)
            self._save(pruned)
            return pruned.get(peer_username)
        except Exception as exc:
            logger.warning(
                "PEER_MEMORY: record_observation(@%s) failed: %s",
                peer_username,
                exc,
            )
            return None

    def format_for_prompt(self, username: str | None, now: float | None = None) -> str:
        """Build the private-context block for one peer.

        Returns ``""`` when peer is unknown or has no useful summary.
        """
        if not username:
            return ""
        summary = self.get_summary(username)
        if summary is None:
            return ""
        return format_for_prompt(summary, now if now is not None else time.time())

    def format_for_prompt_many(
        self,
        usernames: list[str | None],
        *,
        self_username: str | None = None,
        now: float | None = None,
    ) -> str:
        """Compose context blocks for several thread participants.

        Filters out self and dedups in-order. Returns the joined block,
        or ``""`` when no candidate is a known peer.
        """
        ts = now if now is not None else time.time()
        seen: set[str] = set()
        blocks: list[str] = []
        for raw in usernames:
            if not raw or raw == self_username or raw in seen:
                continue
            seen.add(raw)
            block = self.format_for_prompt(raw, ts)
            if block:
                blocks.append(block)
        return "\n\n".join(blocks)


def default_peer_memory_path(self_username: str) -> Path:
    """Default JSON file path keyed by self-username.

    Lives at ``~/.langchain-colony/peer-memory-<self>.json``. Multiple
    agents on the same host don't collide because each writes to its
    own filename.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self_username)
    return PEER_MEMORY_DEFAULT_DIR / f"peer-memory-{safe}.json"
