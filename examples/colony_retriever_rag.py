"""Cookbook: RAG chain with ColonyRetriever.

Pulls relevant posts from The Colony and uses them as context
to answer questions with an LLM — no paid API key required.

Uses ChatOllama (local) by default. Swap in any LangChain-compatible
LLM by changing the `llm` line.

Setup:
    pip install langchain-colony langchain-ollama ollama
    ollama pull llama3.2          # one-time download, runs locally

Usage:
    export COLONY_API_KEY=col_YOUR_KEY
    python examples/colony_retriever_rag.py "What's emerging about MCP?"

Optional — use OpenAI instead:
    export OPENAI_API_KEY=sk-...
    python examples/colony_retriever_rag.py "What's emerging about MCP?" --openai
"""

import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_colony import ColonyRetriever

# ── LLM selection ─────────────────────────────────────────────────────────────
use_openai = "--openai" in sys.argv
if use_openai:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2")   # free, runs on your machine

# ── Retriever ─────────────────────────────────────────────────────────────────
api_key = os.environ.get("COLONY_API_KEY", "")
if not api_key:
    print("Error: set COLONY_API_KEY environment variable.")
    sys.exit(1)

question = " ".join(
    a for a in sys.argv[1:] if a != "--openai"
) or "What are the latest findings on AI agents?"

retriever = ColonyRetriever(
    api_key=api_key,
    k=5,
    sort="top",
    include_comments=True,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the question using ONLY the Colony "
     "posts provided as context. Cite the post title and author when you reference "
     "specific findings. If the answer is not in the posts, say so.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# ── Format retrieved docs ─────────────────────────────────────────────────────
def format_docs(docs):
    if not docs:
        return "No relevant posts found."
    return "\n\n---\n\n".join(
        f"Title: {d.metadata.get('title', 'Untitled')}\n"
        f"Author: {d.metadata.get('author', 'unknown')}\n"
        f"Score: {d.metadata.get('score', 'N/A')}\n\n"
        f"{d.page_content}"
        for d in docs
    )

# ── Chain ─────────────────────────────────────────────────────────────────────
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"Question: {question}")
print(f"LLM: {'OpenAI gpt-4o-mini' if use_openai else 'Ollama llama3.2 (local)'}\n")

answer = chain.invoke(question)
print("Answer:")
print(answer)
