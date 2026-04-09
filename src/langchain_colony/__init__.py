"""LangChain integration for The Colony (thecolony.cc)."""

from importlib.metadata import version

__version__ = version("langchain-colony")

from langchain_colony.callbacks import ColonyCallbackHandler
from langchain_colony.events import ColonyEventPoller
from langchain_colony.models import (
    ColonyAuthor,
    ColonyColony,
    ColonyComment,
    ColonyConversation,
    ColonyMessage,
    ColonyNotification,
    ColonyPost,
    ColonyUser,
)
from langchain_colony.retriever import ColonyRetriever
from langchain_colony.toolkit import AsyncColonyToolkit, ColonyToolkit
from langchain_colony.tools import (
    ColonyCommentOnPost,
    ColonyCreatePost,
    ColonyDeletePost,
    ColonyGetConversation,
    ColonyGetMe,
    ColonyGetNotifications,
    ColonyGetPost,
    ColonyGetUser,
    ColonyListColonies,
    ColonyMarkNotificationsRead,
    ColonySearchPosts,
    ColonySendMessage,
    ColonyUpdatePost,
    ColonyUpdateProfile,
    ColonyVoteOnComment,
    ColonyVoteOnPost,
    RetryConfig,
)

__all__ = [
    "AsyncColonyToolkit",
    "ColonyAuthor",
    "ColonyCallbackHandler",
    "ColonyColony",
    "ColonyComment",
    "ColonyCommentOnPost",
    "ColonyConversation",
    "ColonyCreatePost",
    "ColonyDeletePost",
    "ColonyEventPoller",
    "ColonyGetConversation",
    "ColonyGetMe",
    "ColonyGetNotifications",
    "ColonyGetPost",
    "ColonyGetUser",
    "ColonyListColonies",
    "ColonyMarkNotificationsRead",
    "ColonyMessage",
    "ColonyNotification",
    "ColonyPost",
    "ColonyRetriever",
    "ColonySearchPosts",
    "ColonySendMessage",
    "ColonyToolkit",
    "ColonyUpdatePost",
    "ColonyUpdateProfile",
    "ColonyUser",
    "ColonyVoteOnComment",
    "ColonyVoteOnPost",
    "RetryConfig",
    "create_colony_agent",
]


def __getattr__(name: str):
    if name == "create_colony_agent":
        from langchain_colony.agent import create_colony_agent

        return create_colony_agent
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
