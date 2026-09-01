"""
Unit tests for codec_agent (server-side ToolWatcher + parse_tool_call).

The fixture-driven cases at the bottom load
testdata/tool-watcher-events.json, a copy of the cross-language
conformance fixture in packages/tool-watcher-conformance in the Codec
repo. Every Codec ToolWatcher implementation (C, TypeScript, Python,
Rust, .NET, Java, and this fork) must reproduce that fixture's event
stream exactly, in order. If any of these regress, this fork's
server-side detection stops matching what the other implementations
produce for the same input.

Run:
    pytest -xvs python/sglang/srt/entrypoints/test_codec_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.srt.entrypoints.codec_agent import (
    DEFAULT_REGION_CAP,
    ToolWatcher,
    WatcherEvent,
    make_call_id,
    parse_tool_call,
    split_watcher_events,
)

# Synthetic markers. Small ints chosen for readable assertions.
START = 90
END = 91


def _kinds(events: list[WatcherEvent]) -> list[str]:
    return [e.kind for e in events]


def test_passthrough_then_region_then_passthrough():
    w = ToolWatcher(start_id=START, end_id=END)
    # "hello world <tool_call> foo bar </tool_call> hello !"
    evs = w.feed([0, 1, START, 3, 4, END, 0, 2])
    assert _kinds(evs) == ["passthrough", "region", "passthrough"]
    assert evs[0].ids == (0, 1)
    assert evs[1].ids == (3, 4)  # markers excluded
    assert evs[2].ids == (0, 2)
    assert not w.inside


def test_region_split_across_feeds():
    w = ToolWatcher(START, END)
    # Feed 1: opens region, no close.
    evs = w.feed([0, START, 3])
    assert _kinds(evs) == ["passthrough"]
    assert evs[0].ids == (0,)
    assert w.inside

    # Feed 2: closes region, then more passthrough.
    evs = w.feed([4, END, 1])
    assert _kinds(evs) == ["region", "passthrough"]
    assert evs[0].ids == (3, 4)
    assert evs[1].ids == (1,)
    assert not w.inside


def test_multiple_regions_in_one_feed():
    w = ToolWatcher(START, END)
    evs = w.feed([0, START, 3, END, 1, START, 4, END, 2])
    assert _kinds(evs) == [
        "passthrough", "region", "passthrough", "region", "passthrough",
    ]
    assert evs[1].ids == (3,)
    assert evs[3].ids == (4,)


def test_stray_end_passes_through():
    w = ToolWatcher(START, END)
    evs = w.feed([0, END, 1])
    # End with no preceding start: treated as ordinary token.
    assert _kinds(evs) == ["passthrough"]
    assert evs[0].ids == (0, END, 1)


def test_nested_start_is_dropped_from_body_but_observable():
    w = ToolWatcher(START, END)
    # <tool_call> body1 <tool_call> body2 </tool_call> trailing
    # The nested start is dropped from the region body but still
    # reported as its own event. The first </tool_call> closes the
    # outer region with body [3, 4].
    evs = w.feed([START, 3, START, 4, END, 99])
    assert _kinds(evs) == ["nested_start", "region", "passthrough"]
    assert evs[0].ids == (START,)
    assert evs[1].ids == (3, 4)
    assert evs[2].ids == (99,)


def test_reset_drops_in_flight_region():
    w = ToolWatcher(START, END)
    w.feed([START, 3, 4])
    assert w.inside
    w.reset()
    assert not w.inside
    evs = w.feed([END, 1])
    assert _kinds(evs) == ["passthrough"]
    assert evs[0].ids == (END, 1)


def test_no_decode_path():
    """The watcher operates on uint32 IDs only. It never reads any
    vocab. It never invokes the tokenizer. Feed IDs that have no
    plausible vocab entry; the watcher must still emit them verbatim."""
    w = ToolWatcher(START, END)
    BIG_A = 0xFFFFFF00
    BIG_B = 0xDEADBEEF
    evs = w.feed([12345, BIG_A, START, BIG_B, END, 99999])
    assert _kinds(evs) == ["passthrough", "region", "passthrough"]
    assert evs[0].ids == (12345, BIG_A)
    assert evs[1].ids == (BIG_B,)
    assert evs[2].ids == (99999,)


def test_ordering_matches_defect3_example():
    """[a, S, X, E, b, S, Y, E, c] must produce five ordered events, not
    a flattened (out_ids, completed_regions) two-tuple that loses where
    each region sat relative to the passthrough runs around it."""
    w = ToolWatcher(START, END)
    a, b, c, x, y = 10, 11, 12, 13, 14
    evs = w.feed([a, START, x, END, b, START, y, END, c])
    assert _kinds(evs) == [
        "passthrough", "region", "passthrough", "region", "passthrough",
    ]
    assert evs[0].ids == (a,)
    assert evs[1].ids == (x,)
    assert evs[2].ids == (b,)
    assert evs[3].ids == (y,)
    assert evs[4].ids == (c,)


def test_empty_region_is_reported_not_skipped():
    """A start marker immediately followed by an end marker is a
    complete, real, empty tool call. It must emit a "region" event with
    ids=(), not be skipped."""
    w = ToolWatcher(START, END)
    evs = w.feed([0, START, END, 1])
    assert _kinds(evs) == ["passthrough", "region", "passthrough"]
    assert evs[1].ids == ()


def test_end_emits_truncated_with_finish_reason():
    w = ToolWatcher(START, END)
    evs = w.feed([0, START, 3, 4])
    assert _kinds(evs) == ["passthrough"]
    assert w.inside

    evs = w.end("length")
    assert _kinds(evs) == ["truncated"]
    assert evs[0].ids == (3, 4)
    assert evs[0].finish_reason == "length"
    assert not w.inside

    # A second end() call is a no-op: nothing left in flight.
    assert w.end("length") == []


def test_end_reports_empty_body_when_stream_ends_right_after_start():
    w = ToolWatcher(START, END)
    w.feed([START])
    assert w.inside

    evs = w.end()  # no finish reason known
    assert _kinds(evs) == ["truncated"]
    assert evs[0].ids == ()
    assert evs[0].finish_reason is None


def test_end_outside_region_emits_nothing():
    w = ToolWatcher(START, END)
    w.feed([START, 3, END, 4])
    assert not w.inside
    assert w.end("stop") == []


def test_region_cap_defaults_and_is_settable():
    w = ToolWatcher(START, END)
    assert w.region_cap == DEFAULT_REGION_CAP

    w.set_region_cap(3)
    assert w.region_cap == 3

    # 0 resets to the default region cap.
    w.set_region_cap(0)
    assert w.region_cap == DEFAULT_REGION_CAP

    w2 = ToolWatcher(START, END, region_cap=3)
    assert w2.region_cap == 3


def test_overflow_fires_once_at_cap_then_resyncs_on_end_marker():
    w = ToolWatcher(START, END, region_cap=3)
    evs = w.feed([START, 1, 2, 3, 4, 5, END, 9])
    assert _kinds(evs) == ["overflow", "passthrough"]
    assert evs[0].ids == (1, 2, 3)
    assert evs[1].ids == (9,)
    assert not w.inside


def test_overflow_then_truncated_reports_both():
    w = ToolWatcher(START, END, region_cap=2)
    evs = w.feed([START, 1, 2, 3, 4])
    assert _kinds(evs) == ["overflow"]
    assert evs[0].ids == (1, 2)

    evs = w.end("length")
    assert _kinds(evs) == ["truncated"]
    assert evs[0].ids == (1, 2)
    assert evs[0].finish_reason == "length"


def test_exact_cap_does_not_overflow():
    w = ToolWatcher(START, END, region_cap=3)
    evs = w.feed([START, 1, 2, 3, END])
    assert _kinds(evs) == ["region"]
    assert evs[0].ids == (1, 2, 3)


# ── split_watcher_events ─────────────────────────────────────────────────────


def test_split_watcher_events_folds_ordered_events_for_call_sites():
    w = ToolWatcher(START, END)
    evs = w.feed([10, START, 13, END, 11, START, 14, END, 12])
    out_ids, bodies, nested_starts = split_watcher_events(evs)
    assert out_ids == [10, 11, 12]
    assert bodies == [(13,), (14,)]
    assert nested_starts == 0


def test_split_watcher_events_counts_nested_starts():
    w = ToolWatcher(START, END)
    evs = w.feed([START, 1, START, 2, END, 3])
    out_ids, bodies, nested_starts = split_watcher_events(evs)
    assert out_ids == [3]
    assert bodies == [(1, 2)]
    assert nested_starts == 1


def test_split_watcher_events_folds_overflow_as_a_completion():
    w = ToolWatcher(START, END, region_cap=2)
    evs = w.feed([START, 1, 2, 3, END, 9])
    out_ids, bodies, nested_starts = split_watcher_events(evs)
    assert out_ids == [9]
    assert bodies == [(1, 2)]  # the overflow's capped prefix
    assert nested_starts == 0


# ── parse_tool_call ─────────────────────────────────────────────────────────


def test_parse_tool_call_well_formed():
    body = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    ev = parse_tool_call(body)
    assert ev.name == "get_weather"
    assert ev.arguments_json == body
    assert ev.id is None


def test_parse_tool_call_with_id():
    ev = parse_tool_call('{"name": "search"}', call_id="tc_00000001")
    assert ev.name == "search"
    assert ev.id == "tc_00000001"


def test_parse_tool_call_malformed_json():
    """Malformed JSON: keep raw body so the caller can return an
    'invalid_arguments' error to the model."""
    body = '{"name": "search"'  # unterminated
    ev = parse_tool_call(body)
    assert ev.name is None
    assert ev.arguments_json == body  # raw body preserved


def test_parse_tool_call_empty():
    ev = parse_tool_call("   ")
    assert ev.name is None
    assert ev.arguments_json == ""


def test_parse_tool_call_compact():
    body = '{"name":"f","arguments":{"a":1}}'
    ev = parse_tool_call(body)
    assert ev.name == "f"


def test_parse_tool_call_no_name_key():
    """Body is JSON but doesn't follow the standard tool-call shape.
    We still return arguments_json; name=None. Caller decides what
    to do with it."""
    body = '{"foo": "bar"}'
    ev = parse_tool_call(body)
    assert ev.name is None
    assert ev.arguments_json == body


# ── call id ─────────────────────────────────────────────────────────────────


def test_make_call_id_format():
    assert make_call_id(1) == "tc_00000001"
    assert make_call_id(0xDEADBEEF) == "tc_deadbeef"


def test_make_call_id_scopes_to_request_id_to_avoid_cross_request_collision():
    """Without request_id, two concurrent requests' first tool call both
    format as "tc_00000001" (defect 6). With request_id, they don't."""
    assert make_call_id(1, request_id="req-a") != make_call_id(1, request_id="req-b")
    assert make_call_id(1, request_id="req-a") == make_call_id(1, request_id="req-a")
    assert make_call_id(1, request_id="req-a") == "tc_req-a_00000001"


# ── Fixture-driven conformance cases ──────────────────────────────────────────
#
# testdata/tool-watcher-events.json is a copy of
# packages/tool-watcher-conformance/fixtures/tool-watcher-events.json from
# the Codec repo, the cross-language source of truth for the event
# contract. Every case there runs here too, against this fork's
# ToolWatcher. This file can't silently fall out of sync with it.

_FIXTURE_PATH = Path(__file__).resolve().parent / "testdata" / "tool-watcher-events.json"
_FIXTURE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _normalize(kind: str, ids, finish_reason: str | None):
    entry = {"kind": kind, "ids": list(ids)}
    if kind == "truncated":
        entry["finish_reason"] = finish_reason
    return entry


@pytest.mark.parametrize(
    "case", _FIXTURE["cases"], ids=[c["name"] for c in _FIXTURE["cases"]]
)
def test_fixture_case(case):
    region_cap = (
        case["region_cap"] if case["region_cap"] is not None else DEFAULT_REGION_CAP
    )
    w = ToolWatcher(
        start_id=_FIXTURE["start_id"], end_id=_FIXTURE["end_id"], region_cap=region_cap
    )

    actual = []
    for feed_ids in case["feeds"]:
        for ev in w.feed(feed_ids):
            actual.append(_normalize(ev.kind, ev.ids, ev.finish_reason))
    if case["end"] is not None:
        for ev in w.end(case["end"]["finish_reason"]):
            actual.append(_normalize(ev.kind, ev.ids, ev.finish_reason))

    expected = [
        _normalize(e["kind"], e["ids"], e.get("finish_reason"))
        for e in case["events"]
    ]
    assert actual == expected
