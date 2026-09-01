"""
Server-side agentic primitives for Codec streaming responses.

Two pieces, layered cleanly on top of the wire format defined in
codec_frame.py:

  - ToolWatcher: a uint32-compare state machine that detects delimited
    regions (tool calls, reasoning blocks, vision spans, sandbox runs)
    in the output token stream without ever decoding. feed() returns a
    single ordered event stream (passthrough, region, nested_start,
    overflow) and end() reports a region left open when generation
    finishes. This class follows the same event contract as the
    canonical libcodec / @codecai/web / codecai / Codec.Net
    implementations, checked here against the shared fixture in
    test_codec_agent.py. This class takes resolved marker IDs the
    caller already looked up. The canonical constructors take a
    tokenizer map and marker names.

  - parse_tool_call: when a region completes, render its body through
    the tokenizer, parse as JSON (the convention every chat-tuned
    model in current use follows), and surface name + arguments_json
    on the next frame.

Why server-side: orchestrators don't have to detokenize on every
frame just to scan for marker text. The server already has the
tokenizer. The server already has the IDs. This PR exposes the
detection result directly in the Codec wire format. Clients get
structured tool_call data alongside the raw token stream this way.

Disabled by default. Activated per-request via the `tool_watcher`
field on CompletionRequest / ChatCompletionRequest.

No new external dependencies beyond msgspec (already a core sglang
dependency) and the codec_frame module in this same package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, List, Literal, Optional, Sequence, Tuple

import msgspec

# ---------------------------------------------------------------------------
# Tool-call data model (mirrors openai-style { id, name, arguments } shape)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEvent:
    """One tool call detected in the model's output stream.

    `arguments_json` is the raw JSON string between the start/end markers.
    `name` is parsed from that JSON when the model uses the standard
    `{"name": "...", "arguments": {...}}` shape; otherwise None.
    """

    name: Optional[str]
    arguments_json: str
    id: Optional[str] = None  # server-generated, e.g. "tc_<seq>"

    def to_wire_dict(self) -> dict:
        """Serialise to the dict shape encoded into msgpack frames and
        the protobuf ToolCall message."""
        out: dict = {"arguments_json": self.arguments_json}
        if self.name is not None:
            out["name"] = self.name
        if self.id is not None:
            out["id"] = self.id
        return out


# ---------------------------------------------------------------------------
# Watcher event contract
# ---------------------------------------------------------------------------

WatcherEventKind = Literal[
    "passthrough", "region", "truncated", "overflow", "nested_start"
]

#: Default cap on the number of token IDs buffered inside one open region.
#: 65536 tokens is comfortably above any real tool-call payload while
#: still bounding worst-case per-watcher memory against a client that can
#: make the model emit a start marker without a matching end marker.
DEFAULT_REGION_CAP: Final[int] = 65536


class WatcherEvent(msgspec.Struct, frozen=True):
    """One event emitted by ToolWatcher.feed() / ToolWatcher.end(), in
    stream order.

    kind is one of:
      "passthrough": IDs outside any watched region.
      "region": a complete start..end region, markers excluded. Reported
        even when empty (start immediately followed by end).
      "truncated": emitted only by end(), when the stream finished while
        still inside a region. ids is whatever was buffered, possibly
        empty. finish_reason carries the reason the stream ended. A
        length stop is distinguishable from a malformed emission this way.
      "overflow": the region buffer hit its configured cap. ids is the
        capped prefix; the watcher keeps scanning for the end marker
        without buffering further body tokens.
      "nested_start": a start marker was seen while already inside a
        region. Dropped from the region body, but reported so it isn't
        silently swallowed. ids is the single-element (id,).

    ids is always a fresh tuple, safe to retain across later feed() calls.

    finish_reason is set only on "truncated" events, only when the
    caller passed one to ToolWatcher.end(). It is None otherwise.
    """

    kind: WatcherEventKind
    ids: Tuple[int, ...]
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Watcher state machine
# ---------------------------------------------------------------------------


class ToolWatcher:
    """Stateful watcher for delimited regions in a token-ID stream.

    feed() runs a single-pass scan of the newly-arrived IDs and returns
    every event from that scan in stream order: passthrough runs,
    complete regions, nested starts, and cap overflows. feed() cannot
    know the stream is over. A region still open when generation
    finishes needs an explicit end() call. end() reports that region as
    "truncated" and keeps its buffered content. No in-flight tool call
    disappears without a trace.

    Construct with the resolved start/end marker IDs directly: the
    caller already owns tokenizer lookup (request.tool_watcher fields,
    or resolve_marker_id in the vLLM fork).
    """

    __slots__ = (
        "_start_id",
        "_end_id",
        "_inside",
        "_capped",
        "_region_cap",
        "_region",
    )

    def __init__(
        self,
        start_id: int,
        end_id: int,
        region_cap: int = DEFAULT_REGION_CAP,
    ) -> None:
        self._start_id: int = start_id
        self._end_id: int = end_id
        self._inside: bool = False
        # True once the in-progress region has hit _region_cap and emitted
        # its "overflow" event. While set, body tokens are dropped (not
        # buffered, not re-reported) until the end marker closes the
        # region.
        self._capped: bool = False
        self._region_cap: int = region_cap if region_cap > 0 else DEFAULT_REGION_CAP
        self._region: List[int] = []

    @property
    def start_id(self) -> int:
        return self._start_id

    @property
    def end_id(self) -> int:
        return self._end_id

    @property
    def inside(self) -> bool:
        """True when a region is currently open (start seen, end not yet)."""
        return self._inside

    @property
    def region_cap(self) -> int:
        """Cap on the number of token IDs buffered inside one open region."""
        return self._region_cap

    def set_region_cap(self, cap: int) -> None:
        """Change the region cap. 0 resets to DEFAULT_REGION_CAP."""
        self._region_cap = cap if cap > 0 else DEFAULT_REGION_CAP

    def reset(self) -> None:
        """Drop any in-flight region. Call between conversations so a
        leftover unclosed region from session N doesn't spill into N+1."""
        self._inside = False
        self._capped = False
        self._region = []

    def feed(self, ids: Sequence[int]) -> List[WatcherEvent]:
        """Feed a chunk of token IDs and return every event from this
        chunk, in stream order.

        Single-pass scan, identical state machine to the C, TypeScript,
        and canonical Python implementations: keep them in sync if you
        change one.
        """
        if not isinstance(ids, (list, tuple)):
            ids = list(ids)

        events: List[WatcherEvent] = []
        n = len(ids)
        pt_start = 0

        for i in range(n):
            tok = ids[i]

            if not self._inside:
                if tok == self._start_id:
                    if i > pt_start:
                        events.append(
                            WatcherEvent(
                                kind="passthrough", ids=tuple(ids[pt_start:i])
                            )
                        )
                    self._inside = True
                    self._capped = False
                    self._region = []
                    # pt_start re-anchors when the region closes.
                # else: token continues the passthrough run; no action.
            else:
                if tok == self._end_id:
                    # Region complete. Skipped when the region already
                    # overflowed: that was reported once already, at the
                    # moment the cap was hit.
                    if not self._capped:
                        events.append(
                            WatcherEvent(kind="region", ids=tuple(self._region))
                        )
                    self._region = []
                    self._inside = False
                    self._capped = False
                    pt_start = i + 1
                elif tok == self._start_id:
                    # Nested start: dropped from the region body. Most
                    # models don't nest these markers. Treating an inner
                    # start as a new region would drop the outer content
                    # silently. Reported as its own event. A caller can
                    # see it happened.
                    events.append(WatcherEvent(kind="nested_start", ids=(tok,)))
                elif self._capped:
                    # Already reported "overflow" for this region. Keep
                    # scanning for the end marker without buffering:
                    # memory stays bounded.
                    pass
                elif len(self._region) >= self._region_cap:
                    # Cap hit on this token. Report what's buffered so
                    # far, then stop growing: do not silently truncate.
                    # self._region is deliberately not cleared here: if
                    # the stream then ends without an end marker, end()
                    # reports the same capped content as "truncated".
                    events.append(
                        WatcherEvent(kind="overflow", ids=tuple(self._region))
                    )
                    self._capped = True
                else:
                    self._region.append(tok)

        # Trailing passthrough run, if we ended outside a region.
        if not self._inside and pt_start < n:
            events.append(
                WatcherEvent(kind="passthrough", ids=tuple(ids[pt_start:n]))
            )

        return events

    def end(self, finish_reason: Optional[str] = None) -> List[WatcherEvent]:
        """Signal end of stream. feed() has no way to know the stream is
        over. Call this once you know no more tokens are coming.

        Returns a single "truncated" event carrying whatever was
        buffered (possibly empty) and finish_reason, when a region was
        still open. Returns an empty list otherwise: calling end() on a
        cleanly finished stream is a no-op.
        """
        if not self._inside:
            return []
        ids = tuple(self._region)
        self._region = []
        self._inside = False
        self._capped = False
        return [WatcherEvent(kind="truncated", ids=ids, finish_reason=finish_reason)]


# ---------------------------------------------------------------------------
# Folding one feed() into what a streaming call site needs
# ---------------------------------------------------------------------------


def split_watcher_events(
    events: Sequence[WatcherEvent],
) -> Tuple[List[int], List[Tuple[int, ...]], int]:
    """Fold one feed()'s ordered events into what the streaming call
    sites need: the passthrough ids to forward, the body of every region
    that finished during this feed, and how many nested start markers
    were dropped.

    A capped region never gets a second "region" event when its end
    marker eventually arrives (see ToolWatcher.feed). Its "overflow"
    event is that region's only completion, folded into the same list.
    "truncated" comes only from ToolWatcher.end. It never appears in
    `events` here. Callers handle a region still open at end of stream
    separately.

    The wire format bundles every completed region from one feed() onto
    the frame carrying that feed()'s passthrough ids. That was true
    before this class returned ordered events, back when it returned two
    flat lists. The event order is what makes that bundling correct now.
    """
    out_ids: List[int] = []
    bodies: List[Tuple[int, ...]] = []
    nested_starts = 0
    for ev in events:
        if ev.kind == "passthrough":
            out_ids.extend(ev.ids)
        elif ev.kind == "region" or ev.kind == "overflow":
            bodies.append(ev.ids)
        elif ev.kind == "nested_start":
            nested_starts += 1
    return out_ids, bodies, nested_starts


# ---------------------------------------------------------------------------
# Body → ToolCallEvent
# ---------------------------------------------------------------------------


def parse_tool_call(
    region_body_text: str, *, call_id: Optional[str] = None
) -> ToolCallEvent:
    """Parse the body of a tool-call region (already detokenized) into
    a structured event.

    The convention every chat-tuned model in current use follows:
        { "name": "<function>", "arguments": { ... } }

    We accept both pretty-printed and compact JSON. If parsing fails
    (malformed body, partial JSON, etc.) we still return an event with
    name=None and arguments_json set to the raw body. The caller can
    surface that to the client. The client can then return an
    "invalid_arguments" error to the model.

    Empty and whitespace-only bodies produce an event with name=None
    and arguments_json="". The shape is the same either way. Downstream
    code can still tell these cases apart from a parsed call.
    """
    body = region_body_text.strip()
    if not body:
        return ToolCallEvent(name=None, arguments_json="", id=call_id)

    name: Optional[str] = None
    try:
        parsed: Any = json.loads(body)
        if isinstance(parsed, dict):
            n = parsed.get("name")
            if isinstance(n, str):
                name = n
    except json.JSONDecodeError:
        # Keep the raw body so the caller can decide how to handle it.
        pass

    return ToolCallEvent(name=name, arguments_json=body, id=call_id)


# ---------------------------------------------------------------------------
# Helpers for the serving layer
# ---------------------------------------------------------------------------


def detokenize_region(tokenizer, region_ids: List[int]) -> str:
    """Convenience wrapper around the tokenizer's batch decode that
    skips special tokens. Tool-call body text is pure JSON, with no
    chat template chrome.

    Tokenizer compatibility: works with any HF AutoTokenizer instance
    (which is what sglang's TokenizerManager exposes). We don't import
    transformers directly, to keep this module dependency-free.
    Duck-typing on .decode() is enough.
    """
    return tokenizer.decode(region_ids, skip_special_tokens=True)


def make_call_id(seq_no: int, *, request_id: Optional[str] = None) -> str:
    """Server-generated tool call id.

    `seq_no` alone is only unique within one request: every stream's
    counter starts at 1. Two concurrent requests' first tool call both
    format as "tc_00000001" this way. Pass the request's own id as
    `request_id` to scope the call id to that request. Omitted, the id
    keeps its old sequence-only shape. Existing fixtures stay
    deterministic that way.
    """
    if request_id:
        return f"tc_{request_id}_{seq_no:08x}"
    return f"tc_{seq_no:08x}"
