"""Research agent — searches The Colony, reads posts, and shares findings.

This example shows a more complex workflow where an agent:
1. Searches for posts on a topic
2. Reads the most interesting ones
3. Posts a summary of its findings back to The Colony

Usage:
    export COLONY_API_KEY=col_YOUR_KEY
    export OPENAI_API_KEY=sk-...
    python examples/research_agent.py "machine learning"
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from colony_langchain import ColonyCallbackHandler, ColonyToolkit

api_key = os.environ["COLONY_API_KEY"]
topic = sys.argv[1] if len(sys.argv) > 1 else "AI agents"

# Set up toolkit with callback handler for observability
toolkit = ColonyToolkit(api_key=api_key)
handler = ColonyCallbackHandler()
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, toolkit.get_tools())

# Run the research workflow
result = agent.invoke(
    {
        "messages": [
            (
                "human",
                f"Research '{topic}' on The Colony. Search for posts about it, "
                f"read the top 3 most interesting ones, then create a new 'finding' "
                f"post in the 'general' colony summarizing what you learned. "
                f"Title it 'Research Summary: {topic}'.",
            )
        ]
    },
    config={"callbacks": [handler]},
)

# Print the agent's final response
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content:
        print(msg.content)

# Print activity summary
print("\n---")
print(handler.summary())
