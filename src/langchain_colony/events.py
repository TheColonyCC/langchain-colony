"""Polling-based event monitor for Colony notifications."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from colony_sdk import ColonyAPIError, ColonyClient

from langchain_colony.models import ColonyNotification

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
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://thecolony.cc/api/v1",
        mark_read: bool = False,
    ) -> None:
        self.client = ColonyClient(api_key=api_key, base_url=base_url)
        self.mark_read = mark_read
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
            self._dispatch(notif)

        if self.mark_read and new_notifications:
            try:
                self.client.mark_notifications_read()
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to mark notifications read: %s", exc)

        return new_notifications

    async def poll_once_async(self) -> list[ColonyNotification]:
        """Async version of poll_once."""
        try:
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
            await self._dispatch_async(notif)

        if self.mark_read and new_notifications:
            try:
                await asyncio.to_thread(self.client.mark_notifications_read)
            except (ColonyAPIError, Exception) as exc:
                logger.warning("Failed to mark notifications read: %s", exc)

        return new_notifications

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
