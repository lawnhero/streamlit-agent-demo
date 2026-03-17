"""
AI Agent Architecture Demo — Streamlit
Classroom tool for demonstrating how agent components work.
Now built with LangChain 1.0: create_agent + @tool + InMemorySaver.

Usage:
  1. pip install -r requirements.txt
  2. set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env
  3. streamlit run app.py
"""

import os
import re
import uuid
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS

# ── LangChain 1.0 imports ─────────────────────────────────────────────────────
from langchain.agents import create_agent                  # replaces AgentExecutor / create_react_agent
from langchain.tools import tool                           # @tool decorator
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver      # replaces ConversationBufferMemory
from langchain_core.messages import AIMessage, ToolMessage, AIMessageChunk

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT_BASE = """\
You are Little L — a sharp, no-BS business AI built for Goizueta Business School students.

**Identity:** You think like a McKinsey analyst meets a startup founder. You cut through noise, \
lead with insight, and back claims with data.

**Personality:** Direct, confident, and occasionally witty. You respect the student's time. \
No filler phrases ("Great question!"), no hedging. Say it once, say it well.

**Tone:** Professional but energetic — think boardroom ready with a espresso shot. \
You engage with real-world business context: markets, strategy, ops, finance.

**Style:** Bullet points over walls of text. Lead with the answer, follow with reasoning. \
Use concrete numbers and examples whenever possible.\
"""

# (provider, model_id) pairs — used with langchain_openai / langchain_anthropic
MODELS = {
    "Haiku 4.5 (fast)": ("anthropic", "claude-haiku-4-5-20251001"),
    "GPT-3.5 Turbo":    ("openai",    "gpt-3.5-turbo"),
}

KNOWLEDGE_BASE = [
    {
        "id": "refund",
        "title": "Refund Policy",
        "content": (
            "Our refund policy allows returns within 30 days of purchase for a full refund. "
            "Items must be unused and in original packaging. "
            "Digital products are non-refundable once downloaded."
        ),
        "keywords": ["refund", "return", "money back", "cancel"],
    },
    {
        "id": "shipping",
        "title": "Shipping Information",
        "content": (
            "Standard shipping takes 3–5 business days. "
            "Express shipping (1–2 days) is available for an additional $15. "
            "Free shipping on all orders over $75."
        ),
        "keywords": ["shipping", "delivery", "ship", "express", "how long"],
    },
    {
        "id": "support",
        "title": "Customer Support",
        "content": (
            "Customer support is available Monday–Friday 9 am–6 pm EST. "
            "Typical response time is under 4 hours during business hours. "
            "Weekend support is by email only."
        ),
        "keywords": ["support", "help", "contact", "hours", "customer service"],
    },
    {
        "id": "pricing",
        "title": "Pricing & Plans",
        "content": (
            "Free tier: 1,000 requests/month. "
            "Startup plan: $99/month. "
            "Enterprise plan: $499/month. "
            "Annual billing saves 20%."
        ),
        "keywords": ["pricing", "price", "plan", "cost", "enterprise", "startup", "free tier"],
    },
]

SUGGESTED = [
    ("What did I just ask you?",                   "🧠", "Tests Memory — try OFF then ON"),
    ("What's 847 × 293?",                          "🔢", "Tests Tools — single tool call"),
    ("What's the refund policy?",                  "📚", "Tests RAG — try OFF then ON"),
    ("How much Bitcoin would $1000 buy today?",    "₿",  "Tests Tools — agent chains search + calculate"),
]

# ── Module-level web search cache (query → results) ───────────────────────────
# Allows @tool functions to share structured results with the Streamlit layer
_search_cache: dict[str, list[dict]] = {}


def _get_chunk_text(chunk) -> str:
    """
    Extract plain text from an AIMessageChunk.
    Handles both str content (OpenAI) and list-of-blocks content (Anthropic).
    """
    c = chunk.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            b.get("text", "") for b in c
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# ── Tool definitions (@tool replaces schema dicts) ────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression precisely.
    Use this for any arithmetic calculation.
    Examples: '847 * 293', '(12 + 8) / 5', '1000 / 42153.50'
    """
    expr = (expression
            .replace("×", "*").replace("÷", "/")
            .replace("−", "-").replace("^", "**"))
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\*]+$', expr):
        return f"Error: unsupported characters in expression '{expression}'"
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def _do_web_search(query: str, max_results: int = 4) -> list[dict]:
    """Execute a web search using DuckDuckGo. Returns list of result dicts."""
    q = query.lower()
    is_news = any(w in q for w in [
        "news", "today", "recent", "latest", "current events",
        "what happened", "who won",
    ])
    try:
        with DDGS() as ddgs:
            if is_news:
                raw = list(ddgs.news(query, max_results=max_results))
                return [
                    {
                        "title": r["title"],
                        "body": r["body"],
                        "href": r["url"],
                        "date": r.get("date", "")[:10],
                        "source": r.get("source", ""),
                    }
                    for r in raw
                ]
            else:
                return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def _format_search_results(results: list[dict]) -> str:
    """Serialize web search results into a string for tool result messages."""
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{r['title']}]{' (' + r['date'] + ')' if r.get('date') else ''}\n"
        f"{r['body']}\nSource: {r.get('href', '')}{(' — ' + r['source']) if r.get('source') else ''}"
        for r in results
    )


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information, news, weather, or any real-time data.
    Use when the user asks about today's news, current events, live prices, or
    anything requiring up-to-date information beyond training knowledge.
    Include today's date in the query when relevant.
    """
    today = date.today().strftime("%B %d, %Y")
    full_query = query if any(d in query for d in ["2025", "2026", today[:4]]) else query
    results = _do_web_search(full_query)
    _search_cache[query] = results          # store for inspector panel
    return _format_search_results(results)


# ── Session state defaults ────────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "token_count": 0,
    "last_trace": [],
    "last_system_prompt": "",
    "last_rag_docs": [],
    "use_memory": False,
    "use_tools": False,
    "use_rag": False,
    "last_web_results": [],
    "use_system_prompt": False,
    "system_prompt_base": DEFAULT_SYSTEM_PROMPT_BASE,
    "model_label": "GPT-3.5 Turbo",
    "max_tokens": 1,
    # LangChain 1.0: persistent checkpointer & stable session thread_id
    "checkpointer": InMemorySaver(),
    "session_thread_id": str(uuid.uuid4()),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Core helpers ──────────────────────────────────────────────────────────────

def rag_retrieve(query: str) -> list[dict]:
    q = query.lower()
    return [d for d in KNOWLEDGE_BASE if any(kw in q for kw in d["keywords"])]


def build_system_prompt(rag_docs: list[dict]) -> str:
    """
    Build the full system prompt for this request.
    Combines persona/constraints (from build_persona) with injected context
    (RAG docs, memory notice, tool instructions) — single function replaces
    the previous build_persona() + build_context() pair.
    Returns empty string when System Prompt toggle is OFF.
    """
    s = st.session_state

    if not s.use_system_prompt:
        return ""

    parts = [s.system_prompt_base]
    constraints = []

    if not s.use_memory:
        constraints.append(
            "You have NO memory of previous turns. You only see the current message. "
            "If asked what was said before, say you don't have access to that."
        )
    if not s.use_tools:
        constraints.append(
            "You have NO tools. You cannot search the web. "
            "For current events or live data, say you can't access that."
        )
    if not s.use_rag:
        constraints.append(
            "You have NO access to company documents. "
            "For questions about policies, pricing, or shipping, say you don't have that info."
        )
    if constraints:
        parts.append("\n\n## Constraints\n" + "\n".join(f"- {c}" for c in constraints))

    if s.use_rag:
        if rag_docs:
            chunks = "\n\n".join(f"[{d['title']}]\n{d['content']}" for d in rag_docs)
            parts.append(
                f"\n\n## Retrieved Knowledge Base Chunks\n{chunks}\n\n"
                "Answer using the above retrieved information where relevant."
            )
        else:
            parts.append(
                "\n\n(RAG is enabled but no relevant documents were retrieved for this query.)"
            )

    return "".join(parts)


def build_agent(system_prompt: str):
    """
    Build a LangChain 1.0 agent using create_agent.

    LangChain 1.0 API (from langchain.agents):
      - model object  → model parameter
      - system_prompt → system_prompt parameter
      - tools list    → tools parameter
      - checkpointer  → checkpointer parameter (InMemorySaver)
    """
    s = st.session_state
    provider, model_id = MODELS[s.model_label]

    max_tok = max(s.max_tokens, 256) if s.use_tools else s.max_tokens

    if provider == "openai":
        model = ChatOpenAI(model=model_id, max_tokens=max_tok)
    else:
        model = ChatAnthropic(model=model_id, max_tokens=max_tok)

    tools_list = [web_search, calculator] if s.use_tools else []

    return create_agent(
        model=model,
        tools=tools_list,
        system_prompt=system_prompt if system_prompt else None,
        checkpointer=s.checkpointer,
    )


def build_trace(
    user_msg: str,
    rag_docs: list[dict],
    tool_calls_list: list[dict] | None = None,
) -> list[dict]:
    """Return trace steps for the inspector panel."""
    s = st.session_state
    tool_calls_list = tool_calls_list or []
    msg = user_msg.lower()
    steps: list[dict] = []

    is_math = bool(
        re.search(r"\b\d[\d\s]*[\+\-×\*\/x÷]\s*\d", msg)
        or any(w in msg for w in ["times", "multiply", "divided", "plus", "minus"])
    )
    is_news = any(w in msg for w in ["news", "today", "recent", "latest", "current events"])
    is_doc_q = any(kw in msg for d in KNOWLEDGE_BASE for kw in d["keywords"])
    is_mem_q = bool(re.search(r"what did i (say|ask)|previous|before|last message|remember", msg))

    display_msg = f'"{user_msg[:55]}…"' if len(user_msg) > 55 else f'"{user_msg}"'

    steps.append({"type": "user", "icon": "👤", "label": "User Input",
                  "text": display_msg, "details": []})

    if not s.use_memory and is_mem_q:
        steps.append({"type": "warning", "icon": "❌", "label": "Memory OFF",
                      "text": "Agent cannot recall prior messages", "details": []})

    if s.use_rag:
        if rag_docs:
            titles = ", ".join(d["title"] for d in rag_docs)
            steps.append({"type": "tool", "icon": "📚", "label": "RAG Retrieval",
                          "text": f"{len(rag_docs)} chunk(s) found: {titles}", "details": []})
        else:
            steps.append({"type": "tool", "icon": "📚", "label": "RAG Retrieval",
                          "text": "No matching documents found", "details": []})
    elif is_doc_q:
        steps.append({"type": "warning", "icon": "❌", "label": "RAG OFF",
                      "text": "Cannot retrieve from knowledge base", "details": []})

    if is_news and not s.use_tools:
        steps.append({"type": "warning", "icon": "❌", "label": "Tools OFF",
                      "text": "Cannot search the web for current information", "details": []})

    llm1_details: list[tuple[str, str]] = []
    if s.use_system_prompt:
        llm1_details.append(("📋", "System Prompt: persona + constraints"))
    if s.use_memory:
        n = min(len(s.messages), 6)
        llm1_details.append(("🧠", f"Memory: {n} prior message(s)" if n else "Memory: enabled (no history yet)"))
    if s.use_rag and rag_docs:
        llm1_details.append(("📚", f"RAG: {len(rag_docs)} chunk(s) injected"))
    if s.use_tools:
        llm1_details.append(("🔧", "Tools available: web_search, calculator"))
    llm1_details.append(("👤", f"User: {display_msg}"))

    llm1_label = "LLM Call #1" if tool_calls_list else "LLM Call"
    steps.append({"type": "llm", "icon": "🤖", "label": llm1_label,
                  "text": "Decides whether to call a tool" if tool_calls_list else "",
                  "details": llm1_details})

    for idx, tci in enumerate(tool_calls_list):
        is_calc = tci["name"] == "calculator"
        if is_calc:
            result_details = [("=", tci.get("calc_result", ""))]
            arg_label = "expression"
            injected_label = f"Result: {tci.get('calc_result', '')}"
        else:
            wr = tci.get("web_results", [])
            result_details = [
                (f"{j+1}.", f"{r['title']}{' (' + r['date'] + ')' if r.get('date') else ''}")
                for j, r in enumerate(wr)
            ]
            arg_label = "query"
            injected_label = f"{len(wr)} search result(s) injected"
        steps.append({"type": "tool", "icon": "🔧",
                      "label": f"Tool: {tci['name'].upper()}",
                      "text": f'{arg_label}: "{tci["query"]}"',
                      "details": result_details})
        next_llm_num = idx + 2
        is_last_tool = idx == len(tool_calls_list) - 1
        steps.append({"type": "llm", "icon": "🤖",
                      "label": f"LLM Call #{next_llm_num}",
                      "text": "Generates final answer" if is_last_tool else "Decides next step",
                      "details": [("🔧", injected_label)]})

    steps.append({"type": "output", "icon": "💬", "label": "Output",
                  "text": "Streaming response to user", "details": []})
    return steps


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🤖 Agent Demo")
    st.caption("Toggle components to see how the agent's behavior changes in real time.")

    st.session_state.model_label = st.selectbox(
        "Model", list(MODELS.keys()), index=list(MODELS.keys()).index(st.session_state.model_label)
    )
    tok_col, slider_col = st.columns([1, 2])
    with tok_col:
        st.metric("Tokens used", f"{st.session_state.token_count:,}")
    with slider_col:
        st.session_state.max_tokens = st.slider(
            "Max tokens", min_value=1, max_value=1024,
            value=st.session_state.max_tokens, step=1,
            help="Caps the length of the response. Lower = shorter answers and faster responses.",
        )
        st.caption(f"~{st.session_state.max_tokens // 4 * 3} words max")

    st.toggle("📋 System Prompt", key="use_system_prompt")
    st.caption(
        "✅ System prompt sent with every request" if st.session_state.use_system_prompt
        else "🚫 Bare LLM — no system prompt, no persona or constraints"
    )
    if st.session_state.use_system_prompt:
        with st.expander("✏️ Edit system prompt", expanded=False):
            edited = st.text_area(
                "Base persona (components append below this)",
                value=st.session_state.system_prompt_base,
                height=140,
                key="_sp_editor",
                label_visibility="collapsed",
            )
            c_save, c_reset = st.columns(2)
            if c_save.button("Save", use_container_width=True, key="_sp_save"):
                st.session_state.system_prompt_base = edited
            if c_reset.button("Reset default", use_container_width=True, key="_sp_reset"):
                st.session_state.system_prompt_base = DEFAULT_SYSTEM_PROMPT_BASE

    st.toggle("🧠 Memory", key="use_memory")
    st.caption(
        "✅ Conversation persists via LangGraph MemorySaver" if st.session_state.use_memory
        else "🚫 Agent forgets after every single message"
    )

    st.toggle("🔧 Tools", key="use_tools")
    st.caption(
        "✅ Web search + calculator via @tool + create_agent" if st.session_state.use_tools
        else "🚫 Limited to training knowledge only"
    )

    st.toggle("📚 RAG", key="use_rag")
    st.caption(
        "✅ Relevant company docs retrieved per query" if st.session_state.use_rag
        else "🚫 No access to internal documents"
    )
    if st.session_state.use_rag:
        with st.expander("📖 Knowledge base", expanded=False):
            if st.session_state.last_rag_docs:
                st.caption(f"**Last retrieved** ({len(st.session_state.last_rag_docs)} chunk(s)):")
                for doc in st.session_state.last_rag_docs:
                    st.markdown(f"**📄 {doc['title']}**")
                    st.write(doc["content"])
                    st.caption(f"Keywords: {', '.join(doc['keywords'])}")
                st.divider()
            st.caption("**Full knowledge base:**")
            for doc in KNOWLEDGE_BASE:
                st.markdown(f"**📄 {doc['title']}**")
                st.write(doc["content"])
                st.caption(f"Keywords: {', '.join(doc['keywords'])}")
                st.markdown("---")

    if st.button("↺ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.session_state.last_trace = []
        st.session_state.last_system_prompt = ""
        st.session_state.last_rag_docs = []
        st.session_state.last_web_results = []
        # New session: fresh thread_id + fresh checkpointer
        st.session_state.checkpointer = InMemorySaver()
        st.session_state.session_thread_id = str(uuid.uuid4())
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI Agent Architecture — Live Demo")

active = [
    label for label, key in [
        ("🧠 Memory", "use_memory"), ("🔧 Tools", "use_tools"),
        ("📚 RAG", "use_rag"),
    ]
    if st.session_state[key]
]
if not st.session_state.use_system_prompt:
    st.warning("**System prompt OFF** — LLM has no persona, constraints, or injected context.")
elif active:
    st.success(f"**Active components:** {' · '.join(active)}")
else:
    st.info("**Active:** Bare LLM — no components enabled. Toggle something in the sidebar!")

col_chat, col_inspector = st.columns([3, 2], gap="large")

# ── Inspector column ──────────────────────────────────────────────────────────
with col_inspector:
    st.markdown("### 🔍 Under the Hood")

    with st.container():
        if st.session_state.last_trace and not isinstance(st.session_state.last_trace[0], dict):
            st.session_state.last_trace = []

        if st.session_state.last_trace:
            STEP_COLORS = {
                "user":    "#6c757d",
                "llm":     "#0d6efd",
                "tool":    "#fd7e14",
                "output":  "#198754",
                "warning": "#dc3545",
            }
            html = ['<div style="font-size:0.88em;line-height:1.5">']
            for i, step in enumerate(st.session_state.last_trace):
                is_last = i == len(st.session_state.last_trace) - 1
                color = STEP_COLORS.get(step["type"], "#6c757d")
                icon, label = step["icon"], step["label"]
                text, details = step.get("text", ""), step.get("details", [])

                bubble = (
                    f'<div style="width:26px;height:26px;background:{color};border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'flex-shrink:0;font-size:13px;margin-top:2px">{icon}</div>'
                )
                label_html = (
                    f'<span style="font-weight:700;color:{color};font-size:0.72em;'
                    f'text-transform:uppercase;letter-spacing:.07em">{label}</span>'
                )
                text_html = (
                    f'<div style="color:#444;font-size:0.9em;margin-top:1px">{text}</div>'
                ) if text else ""
                detail_rows = "".join(
                    f'<div style="display:flex;gap:6px;padding:1px 0">'
                    f'<span style="flex-shrink:0">{di}</span>'
                    f'<span style="color:#555">{dt}</span></div>'
                    for di, dt in details
                )
                details_html = (
                    f'<div style="margin-top:5px;border-left:3px solid {color}55;'
                    f'padding:3px 0 3px 10px">{detail_rows}</div>'
                ) if details else ""

                html.append(
                    f'<div style="display:flex;align-items:flex-start;gap:8px;padding:2px 0">'
                    f'{bubble}'
                    f'<div style="flex:1;padding-top:3px">{label_html}{text_html}{details_html}</div>'
                    f'</div>'
                )
                if not is_last:
                    html.append(
                        f'<div style="margin-left:12px;width:2px;height:12px;background:#dee2e6"></div>'
                    )
            html.append('</div>')
            st.markdown("".join(html), unsafe_allow_html=True)
        else:
            st.caption("Step-by-step trace will appear here after your first message.")
            st.markdown(
                "**What you'll see:**\n"
                "- User → LLM → Tool → LLM → Output flow\n"
                "- Each LLM call shows exactly what was in the prompt\n"
                "- Missing-component warnings when toggles are OFF"
            )


# ── Chat column ───────────────────────────────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Chat with Little L")

    st.caption("Suggested questions — each tests a different component:")
    sq1, sq2 = st.columns(2)
    for i, (question, icon, hint) in enumerate(SUGGESTED):
        with (sq1 if i % 2 == 0 else sq2):
            if st.button(f"{icon} {question}", use_container_width=True, help=hint, key=f"sq{i}"):
                st.session_state._pending_q = question

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    stream_slot = st.empty()

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask Little L anything...")

if "_pending_q" in st.session_state:
    prompt = st.session_state.pop("_pending_q")

# ── Process input ─────────────────────────────────────────────────────────────
if prompt:
    rag_docs = rag_retrieve(prompt) if st.session_state.use_rag else []
    system_prompt = build_system_prompt(rag_docs)

    # ── Memory: LangGraph MemorySaver + thread_id ─────────────────────────────
    # Memory ON  → reuse the same session thread_id (agent remembers history)
    # Memory OFF → new UUID each request (agent starts fresh every turn)
    thread_id = (
        st.session_state.session_thread_id
        if st.session_state.use_memory
        else str(uuid.uuid4())
    )
    config = {"configurable": {"thread_id": thread_id}}

    agent = build_agent(system_prompt)

    tool_calls_list: list[dict] = []
    full_response = ""

    input_tokens = len(system_prompt) // 4 + len(prompt) // 4

    provider, _ = MODELS[st.session_state.model_label]

    api_key_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_var, "")

    with stream_slot.container():
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key:
                full_response = f"⚠️ {api_key_var} not set. Add it to .env and restart."
                st.error(full_response)
            else:
                try:
                    # ── LangChain 1.0 streaming ───────────────────────────────
                    # agent.stream(stream_mode="messages") yields (chunk, metadata)
                    # tuples — AIMessageChunks for text/tool decisions,
                    # ToolMessages for tool results.
                    response_placeholder = st.empty()
                    status_placeholder = st.empty()

                    # Track active tool calls (tool call id → tool name string)
                    pending_tool_map: dict[str, str] = {}

                    # ── LangChain 1.0: agent.stream(stream_mode="messages") ──────
                    # Each iteration yields (chunk, metadata).
                    # AIMessageChunk: either text tokens (final response) or a
                    #   tool-call start signal. Args are streamed as partial_json
                    #   content blocks (not in chunk.tool_calls), so we use
                    #   agent.get_state() after streaming for complete args.
                    # ToolMessage: tool finished — show status.
                    for chunk, metadata in agent.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config=config,
                        stream_mode="messages",
                    ):
                        if isinstance(chunk, AIMessageChunk):
                            if chunk.tool_calls:
                                # First chunk of a tool call — name is available here
                                for tc in chunk.tool_calls:
                                    if tc.get("id") and tc.get("name"):
                                        pending_tool_map[tc["id"]] = tc["name"]
                                        status_placeholder.markdown(
                                            f"🔧 Calling **{tc['name']}**..."
                                        )
                            else:
                                # Text response tokens.
                                # _get_chunk_text handles str (OpenAI) and
                                # list-of-content-blocks (Anthropic).
                                text = _get_chunk_text(chunk)
                                if text:
                                    full_response += text
                                    response_placeholder.markdown(full_response + "▌")

                        elif isinstance(chunk, ToolMessage):
                            status_placeholder.markdown(
                                f"🔧 **{pending_tool_map.get(chunk.tool_call_id, 'tool')}** "
                                "done — processing result..."
                            )

                    # After streaming: use get_state() to get complete tool args
                    # (args stream as partial_json blocks; only full state has them)
                    if pending_tool_map and st.session_state.use_tools:
                        final_state = agent.get_state(config)
                        tool_call_id_to_args: dict[str, dict] = {}
                        tool_call_id_to_result: dict[str, str] = {}

                        for msg in final_state.values.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_call_id_to_args[tc["id"]] = {
                                        "name": tc["name"],
                                        "args": tc.get("args", {}),
                                    }
                            elif isinstance(msg, ToolMessage):
                                tool_call_id_to_result[msg.tool_call_id] = msg.content

                        for tc_id, tc_info in tool_call_id_to_args.items():
                            name = tc_info["name"]
                            args = tc_info["args"]
                            result = tool_call_id_to_result.get(tc_id, "")
                            if name == "calculator":
                                tool_calls_list.append({
                                    "name": "calculator",
                                    "query": args.get("expression", ""),
                                    "calc_result": result,
                                    "web_results": [],
                                })
                            elif name == "web_search":
                                query_str = args.get("query", "")
                                tool_calls_list.append({
                                    "name": "web_search",
                                    "query": query_str,
                                    "calc_result": "",
                                    "web_results": _search_cache.get(query_str, []),
                                })

                    # Finalize display
                    status_placeholder.empty()
                    if full_response:
                        response_placeholder.markdown(full_response)
                    else:
                        full_response = "*(no response)*"
                        response_placeholder.markdown(full_response)

                except Exception as exc:
                    full_response = f"⚠️ API error: {exc}"
                    st.error(full_response)

    # Build trace with actual tool call data
    trace = build_trace(prompt, rag_docs, tool_calls_list)
    st.session_state.last_trace = trace
    st.session_state.last_system_prompt = system_prompt
    st.session_state.last_rag_docs = rag_docs
    st.session_state.last_web_results = [
        r for tci in tool_calls_list for r in tci.get("web_results", [])
    ]

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.token_count += input_tokens + len(full_response) // 4

    st.rerun()
