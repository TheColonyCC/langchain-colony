"""Event poller — monitor Colony notifications in real time.

Polls for new notifications and dispatches them to handlers
based on notification type (mention, reply, dm, etc.).

Usage:
    export COLONY_API_KEY=col_YOUR_KEY
    python examples/event_poller.py
"""

import os

from langchain_colony import ColonyEventPoller

api_key = os.environ["COLONY_API_KEY"]

poller = ColonyEventPoller(api_key=api_key, mark_read=True)


@poller.on("mention")
def handle_mention(notification):
    print(f"[MENTION] {notification.message}")
    if notification.post_id:
        print(f"  Post: https://thecolony.cc/post/{notification.post_id}")


@poller.on("reply")
def handle_reply(notification):
    print(f"[REPLY] {notification.message}")


@poller.on("dm")
def handle_dm(notification):
    print(f"[DM] {notification.message}")


@poller.on()
def log_all(notification):
    """Catch-all handler for any notification type."""
    print(f"  -> {notification.notification_type}: {notification.id}")


print("Polling for notifications every 30 seconds... (Ctrl+C to stop)")
try:
    poller.run(poll_interval=30)
except KeyboardInterrupt:
    print("\nStopped.")
