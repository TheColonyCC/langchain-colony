"""Tests for v0.9.0 peer-memory primitives."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_colony.peer_memory import (
    JSONFilePeerMemoryStore,
    PeerObservation,
    PeerSummary,
    VoteHistory,
    apply_observation,
    cap_by_last_seen,
    compute_relationship,
    default_peer_memory_path,
    format_for_prompt,
    new_summary,
    prune_stale,
)

# ──────────────────────────────────────────────────────────────────────
# new_summary + dataclass round-trip
# ──────────────────────────────────────────────────────────────────────


class TestNewSummary:
    def test_default_state(self):
        s = new_summary("alice", 1000.0)
        assert s.username == "alice"
        assert s.first_seen == 1000.0
        assert s.last_seen == 1000.0
        assert s.interaction_count == 0
        assert s.topics == {}
        assert s.vote_history.up == 0
        assert s.vote_history.down == 0
        assert s.style_notes == ""
        assert s.recent_positions == []
        assert s.relationship == "neutral"


class TestPeerSummaryRoundTrip:
    def test_to_dict_then_from_dict(self):
        s = PeerSummary(
            username="bob",
            first_seen=1.0,
            last_seen=2.0,
            interaction_count=4,
            topics={"sec": 3, "ml": 1},
            vote_history=VoteHistory(up=2, down=1),
            style_notes="terse, factual",
            recent_positions=["likes A over B", "skeptical of C"],
            relationship="mixed",
        )
        d = s.to_dict()
        assert d["vote_history"] == {"up": 2, "down": 1}
        s2 = PeerSummary.from_dict(d)
        assert s2 == s

    def test_from_dict_handles_missing_fields(self):
        s = PeerSummary.from_dict({"username": "alice"})
        assert s.username == "alice"
        assert s.first_seen == 0.0
        assert s.interaction_count == 0
        assert s.vote_history.up == 0
        assert s.relationship == "neutral"


# ──────────────────────────────────────────────────────────────────────
# apply_observation
# ──────────────────────────────────────────────────────────────────────


class TestApplyObservation:
    def setup_method(self):
        self.base = new_summary("alice", 1000.0)

    def test_increments_count_and_last_seen(self):
        next_s = apply_observation(self.base, PeerObservation(kind="engagement-comment"), 2000.0)
        assert next_s.interaction_count == 1
        assert next_s.last_seen == 2000.0
        assert next_s.first_seen == 1000.0

    def test_topics_counter_accumulates(self):
        a = apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", topics=["security", "DMs"]),
            2000.0,
        )
        b = apply_observation(
            a,
            PeerObservation(kind="engagement-comment", topics=["security"]),
            3000.0,
        )
        assert b.topics == {"security": 2, "dms": 1}

    def test_topics_skips_empty_strings(self):
        next_s = apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", topics=["", "   ", "real"]),
            2000.0,
        )
        assert next_s.topics == {"real": 1}

    def test_recent_positions_ring_max_3(self):
        s = self.base
        for i, txt in enumerate(["first", "second", "third", "fourth"]):
            s = apply_observation(
                s,
                PeerObservation(kind="engagement-comment", position=txt),
                2000.0 + i,
            )
        assert s.recent_positions == ["fourth", "third", "second"]

    def test_recent_positions_dedup_to_front(self):
        s = apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", position="x"),
            2000.0,
        )
        s = apply_observation(s, PeerObservation(kind="engagement-comment", position="y"), 3000.0)
        s = apply_observation(s, PeerObservation(kind="engagement-comment", position="x"), 4000.0)
        assert s.recent_positions == ["x", "y"]

    def test_position_truncated_to_200_chars(self):
        long_position = "a" * 500
        next_s = apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", position=long_position),
            2000.0,
        )
        assert len(next_s.recent_positions[0]) == 200

    def test_position_whitespace_only_is_ignored(self):
        next_s = apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", position="   "),
            2000.0,
        )
        assert next_s.recent_positions == []

    def test_auto_upvote_increments_up(self):
        next_s = apply_observation(self.base, PeerObservation(kind="auto-upvote"), 2000.0)
        assert next_s.vote_history.up == 1
        assert next_s.vote_history.down == 0

    def test_manual_vote_tallies_as_upvote(self):
        next_s = apply_observation(self.base, PeerObservation(kind="manual-vote"), 2000.0)
        assert next_s.vote_history.up == 1

    def test_auto_downvote_increments_down(self):
        next_s = apply_observation(self.base, PeerObservation(kind="auto-downvote"), 2000.0)
        assert next_s.vote_history.up == 0
        assert next_s.vote_history.down == 1

    def test_non_vote_kind_leaves_vote_history_untouched(self):
        next_s = apply_observation(self.base, PeerObservation(kind="dm-received"), 2000.0)
        assert next_s.vote_history.up == 0
        assert next_s.vote_history.down == 0

    def test_does_not_mutate_input(self):
        before = (self.base.interaction_count, dict(self.base.topics), self.base.recent_positions[:])
        apply_observation(
            self.base,
            PeerObservation(kind="engagement-comment", topics=["x"], position="y"),
            2000.0,
        )
        assert before == (
            self.base.interaction_count,
            dict(self.base.topics),
            self.base.recent_positions[:],
        )


# ──────────────────────────────────────────────────────────────────────
# compute_relationship
# ──────────────────────────────────────────────────────────────────────


class TestComputeRelationship:
    def test_neutral_under_3_interactions(self):
        assert compute_relationship(VoteHistory(up=5, down=0), 2) == "neutral"

    def test_agreed_on_2plus_net_upvotes(self):
        assert compute_relationship(VoteHistory(up=3, down=1), 4) == "agreed"

    def test_disagreed_on_2plus_net_downvotes(self):
        assert compute_relationship(VoteHistory(up=0, down=3), 3) == "disagreed"

    def test_mixed_when_one_of_each(self):
        assert compute_relationship(VoteHistory(up=1, down=1), 5) == "mixed"

    def test_neutral_on_no_votes_after_3_interactions(self):
        assert compute_relationship(VoteHistory(up=0, down=0), 5) == "neutral"

    def test_neutral_on_only_one_upvote(self):
        # 1 net upvote isn't enough for "agreed".
        assert compute_relationship(VoteHistory(up=1, down=0), 3) == "neutral"


# ──────────────────────────────────────────────────────────────────────
# prune_stale + cap_by_last_seen
# ──────────────────────────────────────────────────────────────────────


class TestPruneStale:
    def test_drops_old_entries(self):
        peer_map = {
            "old": PeerSummary(username="old", first_seen=0, last_seen=1000.0),
            "fresh": PeerSummary(username="fresh", first_seen=0, last_seen=9000.0),
        }
        out = prune_stale(peer_map, 5000.0, 10_000.0)
        assert "fresh" in out
        assert "old" not in out

    def test_zero_ttl_is_noop(self):
        peer_map = {"old": PeerSummary(username="old", first_seen=0, last_seen=1000.0)}
        assert prune_stale(peer_map, 0, 99_999_999.0) is peer_map


class TestCapByLastSeen:
    def test_drops_oldest_over_cap(self):
        peer_map = {
            "a": PeerSummary(username="a", first_seen=0, last_seen=1000.0),
            "b": PeerSummary(username="b", first_seen=0, last_seen=2000.0),
            "c": PeerSummary(username="c", first_seen=0, last_seen=3000.0),
        }
        out = cap_by_last_seen(peer_map, 2)
        assert sorted(out.keys()) == ["b", "c"]

    def test_zero_max_is_noop(self):
        peer_map = {"a": PeerSummary(username="a", first_seen=0, last_seen=1.0)}
        assert cap_by_last_seen(peer_map, 0) is peer_map

    def test_below_cap_is_identity(self):
        peer_map = {"a": PeerSummary(username="a", first_seen=0, last_seen=1.0)}
        assert cap_by_last_seen(peer_map, 10) is peer_map


# ──────────────────────────────────────────────────────────────────────
# format_for_prompt
# ──────────────────────────────────────────────────────────────────────


class TestFormatForPrompt:
    def test_zero_interactions_returns_empty(self):
        assert format_for_prompt(new_summary("alice", 1000.0), 2000.0) == ""

    def test_full_render(self):
        s = PeerSummary(
            username="alice",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=5,
            topics={"security": 4, "dms": 2, "ml": 1},
            vote_history=VoteHistory(up=3, down=0),
            style_notes="concrete examples preferred",
            recent_positions=["arguing for X", "skeptical of Y"],
            relationship="agreed",
        )
        block = format_for_prompt(s, 1000.0 + 3 * 86400)
        assert "@alice" in block
        assert "3 days ago" in block
        assert "5 prior interactions" in block
        assert "security, dms, ml" in block
        assert "concrete examples preferred" in block
        assert "arguing for X" in block
        assert "Relationship: agreed" in block

    def test_singular_forms_at_count_1(self):
        s = PeerSummary(
            username="bob",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=1,
        )
        block = format_for_prompt(s, 1000.0 + 86400)
        assert "1 day ago" in block
        assert "1 prior interaction" in block
        assert "1 prior interactions" not in block

    def test_suppresses_style_notes_when_empty(self):
        s = PeerSummary(
            username="alice",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=1,
        )
        assert "Notes:" not in format_for_prompt(s, 1000.0)

    def test_suppresses_topics_when_empty(self):
        s = PeerSummary(
            username="alice",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=1,
        )
        assert "Topics they care about:" not in format_for_prompt(s, 1000.0)

    def test_suppresses_recent_positions_when_empty(self):
        s = PeerSummary(
            username="alice",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=1,
        )
        assert "Recent positions:" not in format_for_prompt(s, 1000.0)

    def test_clamps_age_to_at_least_one_day(self):
        s = PeerSummary(
            username="alice",
            first_seen=1000.0,
            last_seen=1000.0,
            interaction_count=1,
        )
        # Same instant → still renders as "1 day ago".
        assert "1 day ago" in format_for_prompt(s, 1000.0)


# ──────────────────────────────────────────────────────────────────────
# default_peer_memory_path
# ──────────────────────────────────────────────────────────────────────


class TestDefaultPath:
    def test_returns_safe_filename(self, tmp_path: Path):
        # Path itself uses the home dir, so we just check the
        # filename-shape transform works for unsafe usernames.
        unsafe = default_peer_memory_path("bad/user!")
        assert unsafe.name == "peer-memory-bad_user_.json"


# ──────────────────────────────────────────────────────────────────────
# JSONFilePeerMemoryStore
# ──────────────────────────────────────────────────────────────────────


class TestJSONFileStore:
    def test_load_when_file_does_not_exist_returns_empty(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "missing.json")
        assert store.get_map() == {}

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        path = tmp_path / "peers.json"
        store = JSONFilePeerMemoryStore(path)
        s = PeerSummary(
            username="bob",
            first_seen=1.0,
            last_seen=2.0,
            interaction_count=3,
            topics={"a": 1},
            vote_history=VoteHistory(up=1, down=0),
            recent_positions=["xyz"],
            relationship="neutral",
        )
        store.save_map({"bob": s})
        # Fresh store reads from disk.
        store2 = JSONFilePeerMemoryStore(path)
        loaded = store2.get_map()
        assert loaded["bob"] == s

    def test_load_corrupted_json_treats_as_empty(self, tmp_path: Path):
        path = tmp_path / "peers.json"
        path.write_text("{not json", encoding="utf-8")
        store = JSONFilePeerMemoryStore(path)
        assert store.get_map() == {}

    def test_load_when_root_is_not_an_object(self, tmp_path: Path):
        path = tmp_path / "peers.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        store = JSONFilePeerMemoryStore(path)
        assert store.get_map() == {}

    def test_load_skips_malformed_entries(self, tmp_path: Path, caplog):
        path = tmp_path / "peers.json"
        path.write_text(
            json.dumps(
                {
                    "ok": {
                        "username": "ok",
                        "first_seen": 1.0,
                        "last_seen": 2.0,
                    },
                    "missing_required": {"first_seen": 1.0},  # no username
                    "not_a_dict": "garbage",
                }
            ),
            encoding="utf-8",
        )
        store = JSONFilePeerMemoryStore(path)
        loaded = store.get_map()
        assert "ok" in loaded
        assert "missing_required" not in loaded
        assert "not_a_dict" not in loaded

    def test_get_summary_returns_none_for_missing(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        assert store.get_summary("nobody") is None

    def test_get_summary_returns_none_for_empty_username(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        assert store.get_summary("") is None

    def test_record_observation_no_username_is_noop(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        out = store.record_observation(None, PeerObservation(kind="engagement-comment"))
        assert out is None
        assert store.get_map() == {}

    def test_record_observation_self_is_noop(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        out = store.record_observation(
            "alice",
            PeerObservation(kind="engagement-comment"),
            self_username="alice",
        )
        assert out is None
        assert store.get_map() == {}

    def test_record_observation_creates_new_entry(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        out = store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment", topics=["security"]),
            self_username="alice",
            now=1000.0,
        )
        assert out is not None
        assert out.username == "bob"
        assert out.interaction_count == 1
        assert out.topics == {"security": 1}

    def test_record_observation_updates_existing_entry(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            self_username="alice",
            now=1000.0,
        )
        out = store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            self_username="alice",
            now=2000.0,
        )
        assert out.interaction_count == 2
        assert out.last_seen == 2000.0

    def test_distillation_callback_runs_at_kth_interaction(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        calls: list[int] = []

        def cb(summary: PeerSummary) -> str:
            calls.append(summary.interaction_count)
            return "fresh notes"

        for i in range(6):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                distill_every=3,
                distillation_callback=cb,
                self_username="alice",
                now=1000.0 + i,
            )
        # Distillation triggers at interaction_count = 3 and 6.
        assert calls == [3, 6]
        assert store.get_summary("bob").style_notes == "fresh notes"

    def test_distillation_callback_failure_preserves_existing_notes(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")

        # Seed with existing notes.
        for _ in range(4):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                self_username="alice",
            )

        def boom(summary: PeerSummary) -> str:
            raise RuntimeError("model down")

        # 5th interaction triggers distillation under default K=5.
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            distillation_callback=boom,
            self_username="alice",
        )
        # Notes preserved (still empty in this case).
        assert store.get_summary("bob").style_notes == ""

    def test_distillation_returning_blank_preserves_existing(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        for _ in range(4):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                self_username="alice",
            )
        # First seed real notes.
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            distillation_callback=lambda s: "kept",
            self_username="alice",
        )
        # Then run another K=5 cycle where the callback returns blank.
        for _ in range(4):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                self_username="alice",
            )
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            distillation_callback=lambda s: "   ",
            self_username="alice",
        )
        assert store.get_summary("bob").style_notes == "kept"

    def test_distillation_truncates_to_500_chars(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        for _ in range(4):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                self_username="alice",
            )
        long_notes = "x" * 800
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            distillation_callback=lambda s: long_notes,
            self_username="alice",
        )
        assert len(store.get_summary("bob").style_notes) == 500

    def test_distill_every_clamps_invalid_to_default(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        calls: list[int] = []

        for i in range(5):
            store.record_observation(
                "bob",
                PeerObservation(kind="engagement-comment"),
                distill_every=0,  # falsy → defaults to 5
                distillation_callback=lambda s: (calls.append(s.interaction_count), "ok")[1],
                self_username="alice",
                now=1000.0 + i,
            )
        # K defaults to 5 → fires once at count=5.
        assert calls == [5]

    def test_ttl_pruning_runs_on_write(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        # Seed an old entry directly.
        old = PeerSummary(username="old", first_seen=0, last_seen=0)
        store.save_map({"old": old})
        # Force-reload from disk.
        store2 = JSONFilePeerMemoryStore(tmp_path / "p.json")
        store2.record_observation(
            "fresh",
            PeerObservation(kind="engagement-comment"),
            ttl_seconds=1.0,
            self_username="me",
            now=10_000.0,
        )
        # Old entry pruned on write.
        loaded = store2.get_map()
        assert "old" not in loaded
        assert "fresh" in loaded

    def test_max_peers_cap_applied_on_write(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        store.save_map(
            {
                "a": PeerSummary(username="a", first_seen=0, last_seen=1000.0),
                "b": PeerSummary(username="b", first_seen=0, last_seen=2000.0),
            }
        )
        store2 = JSONFilePeerMemoryStore(tmp_path / "p.json")
        store2.record_observation(
            "c",
            PeerObservation(kind="engagement-comment"),
            max_peers=2,
            self_username="me",
            now=3000.0,
        )
        loaded = store2.get_map()
        # Oldest ("a") evicted; "b" + "c" survive.
        assert sorted(loaded.keys()) == ["b", "c"]

    def test_record_observation_swallows_internal_errors(self, tmp_path: Path, monkeypatch):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")

        def boom(*_args, **_kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(store, "_save", boom)
        # Should not raise — non-throwing contract.
        out = store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            self_username="alice",
        )
        assert out is None

    def test_format_for_prompt_unknown_peer_returns_empty(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        assert store.format_for_prompt("stranger") == ""

    def test_format_for_prompt_empty_username_returns_empty(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        assert store.format_for_prompt(None) == ""
        assert store.format_for_prompt("") == ""

    def test_format_for_prompt_known_peer(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment", topics=["sec"]),
            self_username="alice",
            now=1000.0,
        )
        block = store.format_for_prompt("bob", now=1000.0)
        assert "@bob" in block
        assert "1 prior interaction" in block

    def test_format_for_prompt_many_filters_self_and_dedups(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        for who in ("bob", "carol"):
            store.record_observation(
                who,
                PeerObservation(kind="engagement-comment"),
                self_username="alice",
                now=1000.0,
            )
        block = store.format_for_prompt_many(
            ["alice", "bob", "bob", "carol", None],
            self_username="alice",
            now=1000.0,
        )
        assert block.count("@bob") == 1
        assert block.count("@carol") == 1
        assert "@alice" not in block

    def test_format_for_prompt_many_empty_when_nothing_known(self, tmp_path: Path):
        store = JSONFilePeerMemoryStore(tmp_path / "p.json")
        assert store.format_for_prompt_many(["stranger"]) == ""

    def test_atomic_write_via_tmp_replace(self, tmp_path: Path):
        path = tmp_path / "nested" / "p.json"
        store = JSONFilePeerMemoryStore(path)
        store.record_observation(
            "bob",
            PeerObservation(kind="engagement-comment"),
            self_username="alice",
        )
        assert path.exists()
        # Tmp file should be cleaned up by .replace().
        assert not path.with_suffix(".json.tmp").exists()
