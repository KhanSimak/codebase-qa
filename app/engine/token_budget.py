"""
token_budget.py — counting tokens before you pay for them

THE PROBLEM:
  Naive RAG sends every retrieved chunk to the LLM in full. Five 100-line
  functions is easily 2,000 tokens of context for a question that might
  only need 8 relevant lines from each. You pay for all 2,000 every time.

THE FIX, IN TWO PARTS:
  1. Compress each chunk to just the lines relevant to THIS question,
     using a cheap Groq call per chunk, run in PARALLEL (asyncio.gather)
     so 5 chunks compress in ~200ms total instead of ~1000ms sequential.
  2. After compression, still enforce a hard token cap with tiktoken.
     If we're somehow still over budget, drop the lowest-ranked chunks
     rather than silently sending an oversized, expensive prompt.

WHY THIS STILL MATTERS EVEN THOUGH GROQ IS NEAR-FREE:
  llama-3.1-8b-instant is $0.05/$0.08 per 1M tokens — compression calls
  cost fractions of a cent regardless. The budget cap exists for a
  second reason beyond cost: it also caps LATENCY (fewer tokens for the
  model to read and generate) and keeps the free tier's token-per-minute
  rate limit from being eaten by oversized prompts.

WHY TIKTOKEN WHEN WE'RE CALLING GROQ, NOT OPENAI:
  tiktoken's cl100k_base encoding doesn't perfectly match Llama's own
  tokenizer, but it's close enough for BUDGETING purposes (deciding
  whether to keep or drop a chunk) — we don't need exact billing-grade
  precision here, just a consistent, fast way to compare chunk sizes.
  If tiktoken isn't installed, we fall back to a character-count
  approximation (~4 chars per token) rather than crashing.
"""

import asyncio
import logging
from groq import AsyncGroq

logger = logging.getLogger(__name__)
_llm = AsyncGroq()

GROQ_MODEL = "llama-3.1-8b-instant"

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    logger.warning("tiktoken unavailable — falling back to ~4 chars/token estimate")
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


MAX_CONTEXT_TOKENS   = 800   # hard cap on total context sent to the final LLM call
MAX_TOKENS_PER_CHUNK = 150   # budget for each individual compression call



def select_context(chunks: list[dict], max_tokens: int = MAX_CONTEXT_TOKENS):
    chunks = sorted(
      chunks,
      key=lambda c: c.get("rerank_score", c.get("score", 0)),
      reverse=True,
)

    selected = []
    used = 0

    MAX_CHARS_PER_CHUNK = 1500

    for chunk in chunks:
      text = chunk["text"][:MAX_CHARS_PER_CHUNK]

      tokens = count_tokens(text)

      if used + tokens > max_tokens:
       continue

    chunk = {**chunk}
    chunk["text"] = text

    selected.append(chunk)
    used += tokens

    logger.info(
        f"Context budget: {used}/{max_tokens} tokens "
        f"({len(selected)}/{len(chunks)} chunks)"
    )
    print("=" * 60)
    print("SELECTED CONTEXT")

    for c in selected:
     print(
        c["name"],
        c["type"],
        c.get("rerank_score"),
    )

    return selected

def build_prompt(
    question: str,
    chunks: list[dict],
    intent: str,
    ):
    if intent == "find_function":
     task = """
 Identify the class or function that best matches the user's question.
Explain its purpose from the repository context.
"""
    elif intent == "understand_flow":
     task = """
Explain how execution flows between functions and classes.
Describe callers, callees and important interactions.
"""

    elif intent == "find_usage":
     task = """
Explain where this symbol is used and what depends on it.
"""
    elif intent == "debug":
     task = """
Identify the code most likely responsible for the reported issue.
Explain why using only the repository context.
""" 
    else:
     task = """
Answer using only the repository context.
"""


    system_prompt = """
    You are answering questions ONLY from the supplied repository context.

    Rules:

1. Use ONLY the provided context.

2. Never use outside knowledge.

3. Never invent:
   - classes
   - functions
   - methods
   - files
   - APIs

4. The user's wording may not exactly match the repository.

If the retrieved context clearly contains the repository's implementation of the requested concept, explain that relationship.

Only answer "The requested symbol was not found" when none of the retrieved symbols are relevant.

5. Read every retrieved symbol before answering.

Do not answer using only the highest-ranked result.

Combine information from multiple retrieved classes, methods and files whenever together they answer the question.
6. Prefer explaining repository concepts over matching names literally.
Example

Question:
What is HTTPClient?

Retrieved context:

Class Client
Class AsyncClient
Class BaseClient

Good answer:

The repository does not define a class named HTTPClient.

Instead it provides two HTTP client implementations:

- Client — synchronous HTTP client.
- AsyncClient — asynchronous HTTP client.

Both inherit from BaseClient.
Question:
How does connection pooling work?

Retrieved context:

Client
AsyncClient
_transport_for_url
PoolTimeout

Good answer:

Connection pooling is implemented by the Client and AsyncClient transports.

Requests are routed through `_transport_for_url()`, which selects the appropriate transport or connection pool for a URL. If no custom transport matches, the default transport is used. When no connection is available within the configured limits, `PoolTimeout` is raised.
    
    """
    system_prompt += "\n\n" + task

    context = []

    for i, c in enumerate(chunks, start=1):
      context.append(
        f"""
Symbol {i}

Type: {c.get("type")}
Name: {c.get("name")}
File: {c.get("file")}
Lines: {c.get("line_start")}-{c.get("line_end")}

Code:

{c["text"]}
"""
       )

    context = "\n\n".join(context)
    print("=" * 80)
    print("CONTEXT")
    print(context)
    user_msg = f"""
Repository context:

{context}

Question:
{question}

Answer using only the repository context.
If the answer cannot be found in the context, explicitly say so.
Cite the file names and function/class names you used.
"""
    return system_prompt, user_msg
