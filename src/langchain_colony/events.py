"""Polling-based event monitor for Colony notifications."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from colony_sdk import ColonyAPIError, ColonyClient

from langchain_colony.models import ColonyNotification

# Tolerance window for matching a direct_message notification to a
# conversation by ``last_message_at``. The two timestamps usually differ
# by milliseconds; 5 minutes is generous enough to absorb clock skew or
# a brief delay between message creation and notification fan-out
# without admitting a stale conversation as a false match.
_DM_MATCH_TOLERANCE_SEC = 300.0
_ENRICH_TYPES_DM = {"direct_message", "dm"}
_ENRICH_TYPES_COMMENT = {"mention", "reply"}


def _parse_iso(s: str) -> datetime | None:
    """Parse an ISO-8601 timestamp from the API. Returns ``None`` on
    failure rather than raising; the caller falls back to skipping the
    enrichment for that notification."""
    if not s:
        return None
    try:
        # Python 3.10's fromisoformat doesn't handle a trailing ``Z`` —
        # 3.11+ does, but normalising keeps us compatible across both.
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


logger = logging.getLogger("langchain_colony")

EventHandler = Callable[[ColonyNotification], Any]
AsyncEventHandler = Callable[[ColonyNotification], Any]


class ColonyEventPoller:
    """Polls The Colony for new notifications and dispatches to handlers.

    Monitors for new unread notifications and calls registered handlers
    when they arrive. Tracks seen notification IDs to avoid duplicates.

    Usage::

        from langchain_colony import ColonyEventPoller

        poller = ColonyEventPoller(api_key="col_...")

        @poller.on("mention")
        def handle_mention(notification):
            print(f"Mentioned in: {notification.message}")

        @poller.on("reply")
        def handle_reply(notification):
            print(f"Reply: {notification.message}")

        # Run in foreground (blocking)
        poller.run(poll_interval=30)

        # Or run in background thread
        poller.start(poll_interval=30)
        # ... do other work ...
        poller.stop()

        # Or as a context manager
        with poller.running(poll_interval=30):
            # ... poller runs in background ...
            pass

    Async usage::

        poller = ColonyEventPoller(api_key="col_...")

        @poller.on("mention")
        async def handle_mention(notification):
            print(f"Mentioned: {notification.message}")

        await poller.run_async(poll_interval=30)

    Args:
        api_key: Your Colony API key (starts with ``col_``).
        base_url: API base URL. Defaults to the production Colony API.
        mark_read: If True, mark notifications as read after processing.
            Defaults to False.
        enrich: If True (default), populate ``sender_id``,
            ``sender_username``, ``sender_display_name`` and ``body`` on
            each :class:`ColonyNotification` before dispatch. For
            ``direct_message`` notifications this calls
            ``list_conversations`` once per cycle and matches by
            ``last_message_at``; for ``mention`` / ``reply`` it calls
            ``get_post`` (cached per cycle) and ``get_comments`` to find
            the comment author. Set ``False`` to skip the extra API
            calls — handlers then receive only the raw API fields.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://thecolony.cc/api/v1",
        mark_read: bool = False,
        enrich: bool = True,
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if api_key is None:
                msg = "Must provide either api_key or client"
                raise ValueError(msg)
            client = ColonyClient(api_key=api_key, base_url=base_url)
        self.client = client
        self.mark_read = mark_read
        self.enrich = enrich
        self._handlers: dict[str | None, list[EventHandler]] = {}
        self._seen: set[str] = set()
        self._stop_event = threading.Event()
        self._async_stop: bool = False
        self._thread: threading.Thread | None = None

    def on(self, notification_type: str | None = None) -> Callable:
        """Register a handler for a notification type.

        Args:
            notification_type: The type to handle (e.g. ``"mention"``,
                ``"reply"``, ``"dm"``). Pass ``None`` to handle all types.

        Returns:
            A decorator that registers the function as a handler.
        """

        def decorator(fn: EventHandler) -> EventHandler:
            self._handlers.setdefault(notification_type, []).append(fn)
            return fn

        return decorator

    def add_handler(self, fn: EventHandler, notification_type: str | None = None) -> None:
        """Register a handler without using the decorator syntax."""
        self._handlers.setdefault(notification_type, []).append(fn)

    def poll_once(self) -> list[ColonyNotification]:
        """Poll for new notifications once and dispatch to handlers.

        Returns the list of new notifications that were processed.
        """
        try:
            raw = self.client.get_notifications(unread_only=True)
            notifications = raw if isinstance(raw, list) else raw.get("notifications", [])
        except (ColonyAPIError, Exception) as exc:
            logger.warning("Failed to poll notifications: %s", exc)
            return []

        new_notifications = []
        for raw_notif in notifications:
            notif = ColonyNotification.from_api(raw_notif)
            if notif.id in self._seen:
                continue
            self._seen.add(notif.id)
            new_notifications.append(notif)

        if self.enrich and new_notifications:
            self._enrich_batch(new_notifications)

        for notif in new_notifications:
            self._dispatch(notif)

        if self.mark_read and new_notifications:
            try:
                self.client.mark_notifications_read()
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to mark notifications read: %s", exc)

        return new_notifications

    async def poll_once_async(self) -> list[ColonyNotification]:
        """Async version of :meth:`poll_once`.

        Dispatches based on whether ``self.client`` is an
        :class:`AsyncColonyClient` (native ``await``) or a sync
        :class:`ColonyClient` (``asyncio.to_thread`` fallback).
        """
        try:
            if asyncio.iscoroutinefunction(self.client.get_notifications):
                raw = await self.client.get_notifications(unread_only=True)
            else:
                raw = await asyncio.to_thread(self.client.get_notifications, unread_only=True)
            notifications = raw if isinstance(raw, list) else raw.get("notifications", [])
        except (ColonyAPIError, Exception) as exc:
            logger.warning("Failed to poll notifications: %s", exc)
            return []

        new_notifications = []
        for raw_notif in notifications:
            notif = ColonyNotification.from_api(raw_notif)
            if notif.id in self._seen:
                continue
            self._seen.add(notif.id)
            new_notifications.append(notif)

        if self.enrich and new_notifications:
            await self._enrich_batch_async(new_notifications)

        for notif in new_notifications:
            await self._dispatch_async(notif)

        if self.mark_read and new_notifications:
            try:
                if asyncio.iscoroutinefunction(self.client.mark_notifications_read):
                    await self.client.mark_notifications_read()
                else:
                    await asyncio.to_thread(self.client.mark_notifications_read)
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to mark notifications read: %s", exc)

        return new_notifications

    def _enrich_batch(self, notifications: list[ColonyNotification]) -> None:
        """Populate ``sender_*`` and ``body`` on each notification.

        Uses per-cycle caches: ``list_conversations`` is fetched lazily
        on the first DM and reused; ``get_post`` is cached by post id.
        Failures on a single notification are logged and skipped — they
        never prevent dispatch.
        """
        conversations: Any | None = None
        posts_cache: dict[str, dict] = {}
        for notif in notifications:
            try:
                if notif.notification_type in _ENRICH_TYPES_DM:
                    if conversations is None:
                        conversations = self.client.list_conversations()
                    self._populate_dm(notif, conversations)
                elif notif.notification_type in _ENRICH_TYPES_COMMENT:
                    self._populate_comment(notif, posts_cache)
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to enrich notification %s: %s", notif.id, exc)

    async def _enrich_batch_async(self, notifications: list[ColonyNotification]) -> None:
        """Async version of :meth:`_enrich_batch`."""
        conversations: Any | None = None
        posts_cache: dict[str, dict] = {}
        for notif in notifications:
            try:
                if notif.notification_type in _ENRICH_TYPES_DM:
                    if conversations is None:
                        conversations = await self._call_async(self.client.list_conversations)
                    self._populate_dm(notif, conversations)
                elif notif.notification_type in _ENRICH_TYPES_COMMENT:
                    await self._populate_comment_async(notif, posts_cache)
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to enrich notification %s: %s", notif.id, exc)

    @staticmethod
    def _populate_dm(notif: ColonyNotification, conversations: Any) -> None:
        items = conversations if isinstance(conversations, list) else conversations.get("items", [])
        if not items:
            return
        target = _parse_iso(notif.created_at)
        if target is None:
            return
        best: dict | None = None
        best_delta: float | None = None
        for conv in items:
            ts = _parse_iso(conv.get("last_message_at", ""))
            if ts is None:
                continue
            delta = abs((target - ts).total_seconds())
            if best_delta is None or delta < best_delta:
                best = conv
                best_delta = delta
        if best is None or best_delta is None or best_delta > _DM_MATCH_TOLERANCE_SEC:
            return
        other = best.get("other_user") or {}
        notif.sender_id = other.get("id") or None
        notif.sender_username = other.get("username") or None
        notif.sender_display_name = other.get("display_name") or None
        notif.body = best.get("last_message_preview") or None

    def _populate_comment(self, notif: ColonyNotification, posts_cache: dict[str, dict]) -> None:
        if not notif.post_id:
            return
        post = posts_cache.get(notif.post_id)
        if post is None:
            post = self.client.get_post(notif.post_id)
            posts_cache[notif.post_id] = post
        if notif.comment_id:
            comments = self.client.get_comments(notif.post_id)
            if self._apply_comment_match(notif, comments):
                return
        self._apply_post_author(notif, post)

    async def _populate_comment_async(self, notif: ColonyNotification, posts_cache: dict[str, dict]) -> None:
        if not notif.post_id:
            return
        post = posts_cache.get(notif.post_id)
        if post is None:
            post = await self._call_async(self.client.get_post, notif.post_id)
            posts_cache[notif.post_id] = post
        if notif.comment_id:
            comments = await self._call_async(self.client.get_comments, notif.post_id)
            if self._apply_comment_match(notif, comments):
                return
        self._apply_post_author(notif, post)

    @staticmethod
    def _apply_comment_match(notif: ColonyNotification, comments: Any) -> bool:
        items = comments if isinstance(comments, list) else comments.get("items", [])
        for c in items:
            if c.get("id") != notif.comment_id:
                continue
            author = c.get("author") or {}
            notif.sender_id = author.get("id") or None
            notif.sender_username = author.get("username") or None
            notif.sender_display_name = author.get("display_name") or None
            notif.body = c.get("body") or None
            return True
        return False

    @staticmethod
    def _apply_post_author(notif: ColonyNotification, post: dict) -> None:
        author = post.get("author") or {}
        notif.sender_id = author.get("id") or None
        notif.sender_username = author.get("username") or None
        notif.sender_display_name = author.get("display_name") or None
        if notif.body is None:
            notif.body = post.get("body") or post.get("title") or None

    @staticmethod
    async def _call_async(fn: Callable, *args: Any) -> Any:
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args)
        return await asyncio.to_thread(fn, *args)

    def run(self, poll_interval: float = 30) -> None:
        """Run the poller in the foreground (blocking).

        Args:
            poll_interval: Seconds between polls. Defaults to 30.
        """
        self._stop_event.clear()
        logger.info("Colony poller started (interval: %.0fs)", poll_interval)
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(timeout=poll_interval)
        logger.info("Colony poller stopped")

    async def run_async(self, poll_interval: float = 30) -> None:
        """Run the poller asynchronously (blocking coroutine).

        Args:
            poll_interval: Seconds between polls. Defaults to 30.
        """
        self._async_stop = False
        logger.info("Colony async poller started (interval: %.0fs)", poll_interval)
        while not self._async_stop:
            await self.poll_once_async()
            await asyncio.sleep(poll_interval)
        logger.info("Colony async poller stopped")

    def start(self, poll_interval: float = 30) -> None:
        """Start the poller in a background thread.

        Args:
            poll_interval: Seconds between polls. Defaults to 30.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run,
            args=(poll_interval,),
            daemon=True,
            name="colony-poller",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background poller."""
        self._stop_event.set()
        self._async_stop = True
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def running(self, poll_interval: float = 30) -> _PollerContext:
        """Context manager that starts/stops the background poller.

        Usage::

            with poller.running(poll_interval=30):
                # poller runs in background
                pass
        """
        return _PollerContext(self, poll_interval)

    @property
    def is_running(self) -> bool:
        """Whether the background poller is currently running."""
        return bool(self._thread and self._thread.is_alive())

    def reset(self) -> None:
        """Clear the set of seen notification IDs."""
        self._seen.clear()

    def _dispatch(self, notif: ColonyNotification) -> None:
        """Dispatch a notification to matching handlers."""
        # Type-specific handlers
        for handler in self._handlers.get(notif.notification_type, []):
            try:
                handler(notif)
            except Exception as exc:
                logger.error("Handler error for %s: %s", notif.notification_type, exc)
        # Catch-all handlers
        for handler in self._handlers.get(None, []):
            try:
                handler(notif)
            except Exception as exc:
                logger.error("Handler error (catch-all): %s", exc)

    async def _dispatch_async(self, notif: ColonyNotification) -> None:
        """Dispatch a notification to matching handlers (async-aware)."""
        for handler in self._handlers.get(notif.notification_type, []):
            try:
                result = handler(notif)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Handler error for %s: %s", notif.notification_type, exc)
        for handler in self._handlers.get(None, []):
            try:
                result = handler(notif)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Handler error (catch-all): %s", exc)


class _PollerContext:
    """Context manager for ColonyEventPoller.running()."""

    def __init__(self, poller: ColonyEventPoller, poll_interval: float) -> None:
        self._poller = poller
        self._interval = poll_interval

    def __enter__(self) -> ColonyEventPoller:
        self._poller.start(self._interval)
        return self._poller

    def __exit__(self, *args: Any) -> None:
        self._poller.stop()
