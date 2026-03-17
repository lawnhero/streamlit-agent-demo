# Implementing an Agent with LangChain 1.0

## What's New in LangChain 1.0

LangChain 1.0 was officially released in **October 2025** as the first stable major version, with a commitment to no breaking changes until 2.0. The biggest architectural shift is the replacement of `create_react_agent` (from `langgraph.prebuilt`) with a unified `create_agent` abstraction from `langchain.agents`. The framework also introduces a **middleware system** for composable agent control, **standard content blocks** for provider-agnostic output, and a streamlined namespace — with legacy chains moved to `langchain-classic`.[^1][^2]

```bash
pip install -U langchain langchain-openai langgraph
```

> Requires **Python 3.10+** (Python 3.9 support dropped at 1.0).[^2]

***

## Package Namespace

LangChain 1.0 reduces the package surface to essentials:[^3]

| Module | What's Available |
|---|---|
| `langchain.agents` | `create_agent`, `AgentState` |
| `langchain.messages` | Message types, `trim_messages`, content blocks |
| `langchain.tools` | `@tool`, `BaseTool` |
| `langchain.chat_models` | `init_chat_model`, `BaseChatModel` |
| `langchain.embeddings` | `Embeddings`, `init_embeddings` |

***

## 1. Installation & Setup

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
```

***

## 2. Defining Tools

The `@tool` decorator is unchanged — the docstring is the description the LLM uses to decide when to call it:[^3]

```python
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """
    Search internal database for customer or product information.
    Use when the user asks about data, records, or accounts.
    """
    return f"Found records matching '{query}': Alice (ID: 001), Bob (ID: 002)"

@tool
def calculate_metrics(metric_type: str, values: list[float]) -> dict:
    """
    Calculate statistical metrics: 'mean', 'sum', or 'growth_pct'.
    values: list of numbers
    """
    if metric_type == "mean":
        return {"result": round(sum(values) / len(values), 2)}
    elif metric_type == "sum":
        return {"result": sum(values)}
    elif metric_type == "growth_pct":
        growth = ((values[-1] - values) / values) * 100
        return {"result": round(growth, 2)}
    return {"error": f"Unknown metric: {metric_type}"}

@tool
def send_notification(recipient: str, message: str) -> dict:
    """Send a notification or alert to a team member. Requires approval."""
    return {"status": "queued", "recipient": recipient}

tools = [search_database, calculate_metrics, send_notification]
```

***

## 3. System Prompt

### Option A: Static `system_prompt` parameter

The simplest approach — pass a string directly to `create_agent`:[^4][^3]

```python
agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt=(
        "You are a helpful business analytics assistant. "
        "Use tools to retrieve and compute data before answering. "
        "Be concise and data-driven."
    )
)
```

### Option B: Dynamic System Prompt via `@dynamic_prompt` middleware

Use the `@dynamic_prompt` decorator for context-aware, per-request system prompts:[^5][^3]

```python
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class UserContext:
    role: str = "analyst"
    user_name: str = "User"

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    base = f"You are a helpful assistant. Address the user as {ctx.user_name}."
    if ctx.role == "admin":
        return f"{base} You have full access to all tools and data."
    elif ctx.role == "viewer":
        return f"{base} You can only read data — do not modify anything."
    return base

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    middleware=[role_based_prompt],
    context_schema=UserContext
)

# Invoke with runtime context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Show me customer data"}]},
    config={"configurable": {"thread_id": "t1", "context": {"role": "admin", "user_name": "Prof. Gu"}}}
)
```

***

## 4. Memory (Persistence)

Memory in LangChain 1.0 uses LangGraph **checkpointers** passed directly to `create_agent`. The same `thread_id` links all turns of a conversation.[^6][^7]

### In-Memory (Development)

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "session-wen-001"}}

# Turn 1
r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for customer Alice"}]},
    config=config
)
print(r1["messages"][-1].content)

# Turn 2 — agent remembers Turn 1
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What did I ask about?"}]},
    config=config
)
print(r2["messages"][-1].content)
```

### SQLite (Persistent Across Restarts)

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="You are a data analytics assistant.",
    checkpointer=checkpointer
)
```

### PostgreSQL / MongoDB (Production)

```python
# MongoDB
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

with MongoDBSaver.from_conn_string("mongodb://user:pass@host:27017/") as checkpointer:
    agent = create_agent(
        model=model,
        system_prompt="You are a chat bot.",
        checkpointer=checkpointer,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "user-42"}},
    )
```

### Memory Comparison

| Backend | Class | Persistence | Use Case |
|---|---|---|---|
| RAM | `InMemorySaver` | Process lifetime | Dev/testing |
| File | `SqliteSaver` | Across restarts | Local/small-scale |
| Postgres | `PostgresSaver` | Distributed | Production |
| MongoDB | `MongoDBSaver` | Distributed | Production |

***

## 5. Middleware

**Middleware** is the defining feature of LangChain 1.0's `create_agent`. It provides composable hooks into the agent execution loop, replacing the need for raw LangGraph graph customization in most cases.[^2]

### Execution Order of Hooks

```
before_agent → [before_model → LLM call → after_model] (loop) → after_agent
                      ↕
               wrap_model_call / wrap_tool_call
```

| Hook | When it runs | Use cases |
|---|---|---|
| `before_agent` | Once, before execution starts | Load memory, validate input |
| `before_model` | Before each LLM call | Update prompts, trim messages, reroute |
| `wrap_model_call` | Around each LLM call | Intercept/modify requests & responses |
| `wrap_tool_call` | Around each tool call | Intercept/modify tool execution |
| `after_model` | After each LLM response | Validate output, apply guardrails |
| `after_agent` | Once, after execution ends | Save results, cleanup |

`before_*` hooks run first-to-last; `after_*` hooks run in **reverse** (last-to-first).[^8]

### Prebuilt Middleware

LangChain 1.0 ships three ready-to-use middlewares:[^3]

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_database, send_notification],
    middleware=[
        # Redact email addresses before they reach the LLM
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # Block requests containing phone numbers
        PIIMiddleware(
            "phone_number",
            detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
            strategy="block"
        ),
        # Summarize conversation when it exceeds 500 tokens
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger={"tokens": 500}
        ),
        # Require human approval before send_notification executes
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_notification": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        ),
    ]
)
```

### Custom Middleware (Class-based)

Subclass `AgentMiddleware` and implement any hooks you need:[^5][^3]

```python
from dataclasses import dataclass
from typing import Callable, Any
from langchain.agents.middleware import AgentMiddleware, ModelRequest, AgentState
from langchain_openai import ChatOpenAI

@dataclass
class UserContext:
    expertise: str = "beginner"

class ExpertiseRoutingMiddleware(AgentMiddleware):
    """Route to different models/tools based on user expertise level."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any]
    ) -> Any:
        level = request.runtime.context.expertise

        if level == "expert":
            new_model = ChatOpenAI(model="gpt-4o")
            new_tools = [search_database, calculate_metrics, send_notification]
        else:
            new_model = ChatOpenAI(model="gpt-4o-mini")
            new_tools = [search_database, calculate_metrics]

        return handler(request.override(model=new_model, tools=new_tools))

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    middleware=[ExpertiseRoutingMiddleware()],
    context_schema=UserContext
)
```

### Custom Middleware (Decorator-based)

For lightweight hooks, use decorators instead of full class inheritance:[^9][^8]

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@before_model(can_jump_to=["end"])
def enforce_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Hard-stop after 50 messages to prevent runaway loops."""
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached. Please start a new session.")],
            "jump_to": "end"
        }
    return None  # Continue normally

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    print(f"[LOG] Agent responded: {last.content[:100]}...")
    return None

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    middleware=[enforce_message_limit, log_response]
)
```

***

## 6. Structured Output

LangChain 1.0 integrates structured output **directly into the main loop**, eliminating an extra LLM call. Use `response_format` with a Pydantic model:[^2][^3]

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class AnalysisResult(BaseModel):
    summary: str
    key_metric: float
    recommendation: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_database, calculate_metrics],
    response_format=ToolStrategy(AnalysisResult)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze sales metrics for the north region"}]
})

structured = result["structured_response"]
print(structured.summary)         # "North region outperformed..."
print(structured.key_metric)      # 120000.0
print(structured.recommendation)  # "Increase inventory..."
```

***

## 7. Standard Content Blocks

New in 1.0: `.content_blocks` provides unified access to reasoning traces, citations, and tool calls across all providers:[^1][^2]

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-6")
response = model.invoke("Explain quantum entanglement.")

for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(f"Reasoning: {block['reasoning']}")
    elif block["type"] == "text":
        print(f"Answer: {block['text']}")
    elif block["type"] == "tool_call":
        print(f"Tool: {block['name']}({block['args']})")
```

***

## 8. Full Working Agent

Combining all components — model, system prompt, tools, memory, middleware, and streaming:

```python
import os
import sqlite3
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.runtime import Runtime
from langchain.agents.middleware import AgentState
from typing import Any

# Tools
@tool
def get_sales(region: str) -> dict:
    """Fetch sales data for a region: north, south, east, or west."""
    data = {"north": 120000, "south": 95000, "east": 140000, "west": 88000}
    return {"region": region, "sales": data.get(region.lower(), 0)}

@tool
def compute_average(numbers: list[float]) -> float:
    """Calculate arithmetic mean of a list of numbers."""
    return round(sum(numbers) / len(numbers), 2)

# Custom middleware: input guard
@before_model(can_jump_to=["end"])
def block_off_topic(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_msg = state["messages"][-1].content.lower()
    if any(kw in last_msg for kw in ["politics", "religion"]):
        return {
            "messages": [AIMessage("I can only assist with business analytics topics.")],
            "jump_to": "end"
        }
    return None

# Persistent memory
conn = sqlite3.connect("analytics_agent.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Build the agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_sales, compute_average],
    system_prompt=(
        "You are a business analytics assistant for Goizueta Business School. "
        "Use tools to retrieve data and compute metrics before answering. "
        "Be concise and data-driven."
    ),
    checkpointer=checkpointer,
    middleware=[
        block_off_topic,
        SummarizationMiddleware(model="gpt-4o-mini", trigger={"tokens": 800}),
    ]
)

# Multi-turn conversation
config = {"configurable": {"thread_id": "wen-analytics-001"}}

queries = [
    "What are north and east region sales?",
    "Calculate the average of those two figures.",
    "Which region performed better, based on what we discussed?",
]

for query in queries:
    print(f"\n>> {query}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    )
    print(f"<< {response['messages'][-1].content}")
```

***

## 9. Streaming

`create_agent` supports token-level streaming out of the box:[^3]

```python
config = {"configurable": {"thread_id": "stream-session"}}

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze east region sales"}]},
    config=config,
    stream_mode="values"
):
    # Each chunk is an agent state update
    last = chunk["messages"][-1]
    if hasattr(last, "content") and last.content:
        print(last.content, end="", flush=True)
```

***

## 10. Migrating from v0.3

| v0.3 | v1.0 |
|---|---|
| `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| `state_modifier=` | `system_prompt=` |
| Manual `StateGraph` for control flow | `middleware=[]` on `create_agent` |
| `ConversationBufferMemory` (deprecated) | `checkpointer=` parameter |
| `AgentExecutor` | Built-in loop in `create_agent` |
| `langchain.retrievers` | `langchain-classic` |
| `langchain.chains.*` | `langchain-classic` |

For backwards compatibility, install `langchain-classic`:[^2]

```bash
pip install langchain-classic
# then update imports:
# from langchain.chains import ... → from langchain_classic.chains import ...
```

***

## 11. Observability (LangSmith)

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-langchain-1.0-agent"
```

All agent invocations are automatically traced in the LangSmith dashboard with full message chains, middleware hooks, tool calls, and token usage.

---

## References

1. [LangChain 1.0 now generally available - Changelog](https://changelog.langchain.com/announcements/langchain-1-0-now-generally-available) - LangChain 1.0 is our first major stable release, marking our commitment to no breaking changes until...

2. [LangChain and LangGraph Agent Frameworks Reach v1.0 Milestones](https://blog.langchain.com/langchain-langgraph-1dot0/) - We're releasing LangChain 1.0 and LangGraph 1.0 — our first major versions of our open source framew...

3. [What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1) - What's new in LangChain v1 ; Persistence. Conversations automatically persist across sessions with b...

4. [What's new in v1 - Docs by LangChain](https://xh-cadd36d0.mintlify.app/oss/javascript/releases/langchain-v1) - What's new in v1 · Main loop integration: Structured output is now generated in the main loop instea...

5. [LangChain Middleware v1-Alpha: A Comprehensive Guide to Agent ...](https://colinmcnamara.com/blog/langchain-middleware-v1-alpha-guide) - The middleware system in LangChain 1.0 alpha operates by modifying the fundamental agent loop throug...

6. [How save chat history on database like mongodb in LangChain 1.0.4](https://forum.langchain.com/t/how-save-chat-history-on-database-like-mongodb-in-langchain-1-0-4/2203) - If you're using LangGraph agents ( create_agent ), add a checkpointer to persist the whole conversat...

7. [LangChain in Chains #54: Create Agents - Stackademic](https://blog.stackademic.com/langchain-in-chains-54-create-agents-32362eaca10f) - You can continue a conversation or agent process over multiple calls by specifying the same thread_i...

8. [Custom middleware - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/middleware/custom) - Both node-style and wrap-style hooks can update agent state. The mechanism differs: Node-style hooks...

9. [Middleware | LangChain Reference](https://reference.langchain.com/python/langchain/middleware/) - Unified reference documentation for LangChain and LangGraph Python packages.

