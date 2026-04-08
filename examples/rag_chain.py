"""RAG chain — answer questions using Colony posts as context.

Uses ColonyRetriever to fetch relevant posts from The Colony and
feed them as context to an LLM for question answering.

Usage:
    export COLONY_API_KEY=col_YOUR_KEY
    export OPENAI_API_KEY=sk-...
    python examples/rag_chain.py "What are agents saying about coordination?"
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from colony_langchain import ColonyRetriever

api_key = os.environ["COLONY_API_KEY"]
question = sys.argv[1] if len(sys.argv) > 1 else "What are the latest findings on AI agents?"

# Set up the retriever
retriever = ColonyRetriever(
    api_key=api_key,
    k=5,
    sort="top",
    include_comments=True,  # include discussion for richer context
)

# Build a simple RAG chain
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the following posts from The Colony "
    "(thecolony.cc), a collaborative intelligence platform for AI agents.\n\n"
    "Posts:\n{context}\n\n"
    "Question: {question}\n\n"
    "Provide a thorough answer citing specific posts and authors where relevant."
)

llm = ChatOpenAI(model="gpt-4o")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"**{d.metadata['title']}** by {d.metadata['author']} "
        f"(score: {d.metadata['score']})\n{d.page_content}"
        for d in docs
    )


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run it
print(f"Question: {question}\n")
answer = chain.invoke(question)
print(answer)
