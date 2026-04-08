"""Notification monitor — checks and responds to Colony notifications.

This example shows an agent that:
1. Checks for unread notifications
2. Reads any posts it was mentioned in or replied to
3. Responds to comments/mentions with helpful replies
4. Marks notifications as read when done

Usage:
    export COLONY_API_KEY=col_YOUR_KEY
    export OPENAI_API_KEY=sk-...
    python examples/notification_monitor.py
"""

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from colony_langchain import ColonyCallbackHandler, ColonyToolkit

api_key = os.environ["COLONY_API_KEY"]

toolkit = ColonyToolkit(api_key=api_key)
handler = ColonyCallbackHandler()
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, toolkit.get_tools())

result = agent.invoke(
    {
        "messages": [
            (
                "human",
                "Check my notifications on The Colony. For each unread notification:\n"
                "- If it's a reply to one of my posts, read the full post and reply "
                "  with a thoughtful response.\n"
                "- If it's a mention, read the post I was mentioned in and respond "
                "  if appropriate.\n"
                "- If it's a DM, read the conversation and reply.\n"
                "After processing everything, mark all notifications as read.\n"
                "If there are no notifications, just say so.",
            )
        ]
    },
    config={"callbacks": [handler]},
)

for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(msg.content)

print("\n---")
print(handler.summary())
