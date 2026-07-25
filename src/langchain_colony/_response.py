"""Normalising a Colony list response to a list, without failing silently.

Why this module exists
---------------------
Colony's list endpoints do not all return the same shape, and this package had
been guessing per call site. Measured against the live API on 2026-07-25:

    get_notifications()   -> bare list
    list_conversations()  -> bare list
    get_colonies()        -> bare list
    get_webhooks()        -> bare list
    get_all_comments()    -> bare list
    get_comments()        -> {"items", "total", "next_cursor", "page"}   <- a REAL envelope

So both shapes genuinely occur, and `get_comments` paginating under ``items`` is
not a legacy artefact — it is current. The per-site guesswork was the problem,
not the tolerance.

The bug this fixes
------------------
Every call site used the shape ``payload if isinstance(payload, list) else
payload.get("<something>", [])``, with ``<something>`` guessed per site:
``notifications``, ``colonies``, ``webhooks``, ``items``. Before colony-sdk
1.30.0, ``AsyncColonyClient`` wrapped every bare-array body as ``{"data": [...]}``
to satisfy a ``-> dict`` annotation on ``_raw_request``. That envelope matched
**none** of the guessed keys, so on the async client:

    raw = {"data": [ ...50 notifications... ]}
    raw if isinstance(raw, list) else raw.get("notifications", [])   ->   []

`ColonyEventPoller.apoll_once()` therefore reported **zero notifications on every
call**, and the async event stream never fired. Nothing raised, nothing logged:
an empty list is a completely plausible answer to "any new notifications?", which
is why this survived several releases. It is the failure mode where the wrong
answer is the reassuring one.

Two changes, and the second matters more than the first
------------------------------------------------------
1. One helper, so the accepted keys are declared in a single place rather than
   guessed per call site.
2. **An unrecognised shape is logged.** Previously it was silently ``[]``, which
   is indistinguishable from "there is nothing here" — so a future server change
   would present as a quiet feature outage rather than a diagnosable fault.

Deliberately a warning and not an exception: these are polling and formatting
paths, and taking down an agent's event loop over an unexpected response shape
is worse than continuing with a loud log. But silence is what let this live, so
silence is what had to go.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("langchain_colony")

# Keys under which the API is known or plausibly expected to nest a list.
#
# ``items`` is REAL and current — ``get_comments()`` paginates under it.
#
# ``data`` is here for one specific reason: it was ``AsyncColonyClient``'s own
# wrapping before colony-sdk 1.30.0, and its absence from the per-site guesses is
# precisely what silently emptied every async list call. The async extra now pins
# ``>=1.30.0`` so the SDK cannot send it, but anyone pinning around that gets
# working behaviour rather than an empty feed — and the tolerance is named here
# rather than implied by an ``else`` branch.
#
# The remainder are the per-endpoint keys the previous call sites guessed at.
# Measurement says the API does not currently send them; they are retained
# because tolerating an unused key is free, whereas removing one that turns out
# to be real re-introduces exactly this bug.
_ENVELOPE_KEYS: tuple[str, ...] = (
    "items",
    "data",
    "notifications",
    "colonies",
    "webhooks",
    "comments",
    "posts",
    "messages",
    "conversations",
    "results",
)


def as_list(payload: Any, context: str) -> list:
    """Return the list carried by ``payload``, or ``[]`` — loudly, if unexpected.

    Args:
        payload: A Colony list response: a bare list, or a dict nesting one
            under a known key.
        context: What was being read, for the log line — e.g.
            ``"get_notifications"``. Named so an operator can tell *which*
            call returned a shape this package did not understand.

    Returns:
        The list. ``[]`` for an empty result, and ``[]`` with a warning logged
        for a shape that carries no recognised list.
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in _ENVELOPE_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        logger.warning(
            "%s returned a dict carrying no recognised list (keys: %s). "
            "Treating it as empty, which may be wrong — if the API added an "
            "envelope, add its key to langchain_colony._response._ENVELOPE_KEYS.",
            context,
            sorted(payload)[:10],
        )
        return []

    logger.warning(
        "%s returned %s, expected a list or a dict enveloping one. Treating it as empty.",
        context,
        type(payload).__name__,
    )
    return []
