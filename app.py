"""
AI Agent Architecture Demo — Streamlit
Classroom tool for demonstrating how agent components work.

Usage:
  1. pip install -r requirements.txt
  2. set ANTHROPIC_API_KEY=your_key   (Windows) or export ANTHROPIC_API_KEY=...
  3. streamlit run app.py
"""

import os
import re
import json
from datetime import date
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT_BASE = (
    "You are 'Little L', a helpful assistant in a business school classroom "
    "demo about AI agent architecture."
)

MODELS = {
    "Haiku 4.5 (fast)": "claude-haiku-4-5-20251001",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
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
            "Express shipping (1–2 days) is available for an additional \$15. "
            "Free shipping on all orders over \$75."
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
            "Startup plan: \$99/month. "
            "Enterprise plan: \$499/month. "
            "Annual billing saves 20%."
        ),
        "keywords": ["pricing", "price", "plan", "cost", "enterprise", "startup", "free tier"],
    },
]

# ── Tool schemas ──────────────────────────────────────────────────────────────
def _web_search_description() -> str:
    today = date.today().strftime("%B %d, %Y")
    return (
        f"Search the internet for current information, news, weather, or any real-time data. "
        f"Today's date is {today}. When the user asks about today's news or current events, "
        f"include the current date in the query (e.g. 'news {today}')."
    )


_CALCULATOR_OPENAI = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression precisely. Use this for any arithmetic calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '847 * 293'",
                },
            },
            "required": ["expression"],
        },
    },
}

_CALCULATOR_ANTHROPIC = {
    "name": "calculator",
    "description": "Evaluate a mathematical expression precisely. Use this for any arithmetic calculation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '847 * 293'",
            },
        },
        "required": ["expression"],
    },
}


def build_openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": _web_search_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string"},
                    },
                    "required": ["query"],
                },
            },
        },
        _CALCULATOR_OPENAI,
    ]


def build_anthropic_tools() -> list[dict]:
    return [
        {
            "name": "web_search",
            "description": _web_search_description(),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"},
                },
                "required": ["query"],
            },
        },
        _CALCULATOR_ANTHROPIC,
    ]

SUGGESTED = [
    ("What did I just ask you?",                   "🧠", "Tests Memory — try OFF then ON"),
    ("What's 847 × 293?",                          "🔢", "Tests Tools — single tool call"),
    ("What's the refund policy?",                  "📚", "Tests RAG — try OFF then ON"),
    ("How much Bitcoin would \$1000 buy today?",    "₿",  "Tests Tools — search then calculate"),
]

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
    "use_workflow": False,
    "last_web_results": [],
    "use_system_prompt": False,
    "system_prompt_base": DEFAULT_SYSTEM_PROMPT_BASE,
    "model_label": "GPT-3.5 Turbo",
    "max_tokens": 1,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Core helpers ──────────────────────────────────────────────────────────────

def rag_retrieve(query: str) -> list[dict]:
    q = query.lower()
    return [d for d in KNOWLEDGE_BASE if any(kw in q for kw in d["keywords"])]


def calculate(expression: str) -> str:
    """Safely evaluate a math expression. Returns the result as a string."""
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


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Execute a web search. For news-like queries uses ddgs.news(), otherwise ddgs.text()."""
    q = query.lower()
    is_news = any(w in q for w in ["news", "today", "recent", "latest", "current events", "what happened", "who won"])
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


def format_search_results(results: list[dict]) -> str:
    """Serialize web search results into a string for tool result messages."""
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{r['title']}]{' (' + r['date'] + ')' if r.get('date') else ''}\n"
        f"{r['body']}\nSource: {r.get('href', '')}{(' — ' + r['source']) if r.get('source') else ''}"
        for r in results
    )


def build_persona() -> str:
    """Persona + constraints block. Returns empty string when system prompt toggle is OFF."""
    if not st.session_state.use_system_prompt:
        return ""
    s = st.session_state
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
    return "".join(parts)


def build_context(history: list[dict], rag_docs: list[dict]) -> str:
    """Injected data block (RAG, memory, tool instructions). Always built when toggles are ON."""
    s = st.session_state
    parts: list[str] = []

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

    if s.use_memory and history:
        hist_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        )
        parts.append(
            f"\n\n## Conversation History (last {min(len(history), 6)} turns)\n{hist_text}"
        )

    if s.use_tools:
        parts.append(
            "\n\n## Calculator\n"
            "- **calculator(expression)**: Evaluate math precisely. "
            "Say 'Calculating: [expr] = [result]' when using it."
        )

    if s.use_workflow:
        parts.append(
            "\n\n## Multi-step Workflow\n"
            "When a question requires real-world data AND a calculation (e.g. 'how much X can I buy for $Y today?'), "
            "first call web_search to get the current value, then call calculator with the result. "
            "For pure arithmetic, call calculator directly."
        )

    return "".join(parts)


def build_trace(
    user_msg: str,
    rag_docs: list[dict],
    tool_calls: list[dict] | None = None,
) -> list[dict]:
    """Return trace steps. tool_calls is the list of tool executions from the agentic loop."""
    s = st.session_state
    tool_calls = tool_calls or []
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

    # ── Workflow hint (shown before LLM call when workflow toggle is ON) ────────
    if s.use_workflow and tool_calls:
        route = " → ".join(tci["name"] for tci in tool_calls)
        steps.append({"type": "workflow", "icon": "⚡", "label": "Workflow",
                      "text": f"Multi-step plan: {route}", "details": []})

    # ── Pre-LLM context gathering ─────────────────────────────────────────────
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

    # ── LLM Call #1 ──────────────────────────────────────────────────────────
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

    # Label the first LLM call — numbered only when tool calls follow
    llm1_label = "LLM Call #1" if tool_calls else "LLM Call"
    steps.append({"type": "llm", "icon": "🤖", "label": llm1_label,
                  "text": "Decides whether to call a tool" if tool_calls else "",
                  "details": llm1_details})

    # ── Interleaved tool executions + follow-up LLM calls ────────────────────
    for idx, tci in enumerate(tool_calls):
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
        is_last_tool = idx == len(tool_calls) - 1
        steps.append({"type": "llm", "icon": "🤖",
                      "label": f"LLM Call #{next_llm_num}",
                      "text": "Generates final answer" if is_last_tool else "Decides next step",
                      "details": [("🔧", injected_label)]})

    steps.append({"type": "output", "icon": "💬", "label": "Output",
                  "text": "Streaming response to user", "details": []})
    return steps


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("# 🤖 Agent Demo")
    with c2:
        st.metric("Tokens", f"{st.session_state.token_count:,}")
    st.caption("Toggle components to see how the agent's behavior changes in real time.")

    st.session_state.model_label = st.selectbox(
        "Model", list(MODELS.keys()), index=list(MODELS.keys()).index(st.session_state.model_label)
    )
    st.session_state.max_tokens = st.slider(
        "Max tokens to generate", min_value=1, max_value=1024,
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
        "✅ History injected into each prompt" if st.session_state.use_memory
        else "🚫 Agent forgets after every single message"
    )

    st.toggle("🔧 Tools", key="use_tools")
    st.caption(
        "✅ Web search + calculator via Function Calling" if st.session_state.use_tools
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

    st.toggle("⚡ Workflow Logic", key="use_workflow")
    st.caption(
        "✅ Multi-step plan shown: search → calculator when needed" if st.session_state.use_workflow
        else "🚫 No workflow guidance — LLM decides on its own"
    )

    if st.button("↺ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.session_state.last_trace = []
        st.session_state.last_system_prompt = ""
        st.session_state.last_rag_docs = []
        st.session_state.last_web_results = []
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🤖 AI Agent Architecture — Live Demo")

active = [
    label for label, key in [
        ("🧠 Memory", "use_memory"), ("🔧 Tools", "use_tools"),
        ("📚 RAG", "use_rag"), ("⚡ Workflow", "use_workflow"),
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
        # Clear stale traces from older app versions (non-dict format)
        if st.session_state.last_trace and not isinstance(st.session_state.last_trace[0], dict):
            st.session_state.last_trace = []

        if st.session_state.last_trace:
            STEP_COLORS = {
                "user":    "#6c757d",
                "llm":     "#0d6efd",
                "tool":    "#fd7e14",
                "workflow":"#e6a817",
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
                "- Workflow bypasses and missing-component warnings"
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

    # Render existing history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Placeholder where streaming output will appear
    stream_slot = st.empty()

# ── Chat input (Streamlit always renders this at page bottom) ─────────────────
prompt = st.chat_input("Ask Little L anything...")

if "_pending_q" in st.session_state:
    prompt = st.session_state.pop("_pending_q")

# ── Process input ─────────────────────────────────────────────────────────────
if prompt:
    rag_docs = rag_retrieve(prompt) if st.session_state.use_rag else []
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    system_prompt = (build_persona() + build_context(history, rag_docs)).strip()

    if st.session_state.use_memory:
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
        ]
    else:
        api_messages = []
    api_messages.append({"role": "user", "content": prompt})

    input_tokens = len(system_prompt) // 4 + sum(len(m["content"]) // 4 for m in api_messages)

    model_id = MODELS[st.session_state.model_label]
    is_openai = model_id.startswith("gpt")

    # These are populated during the API interaction below
    tool_calls_list: list[dict] = []   # one entry per tool execution
    full_response: str = ""

    with stream_slot.container():
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            if is_openai:
                openai_key = os.environ.get("OPENAI_API_KEY", "")
                if not openai_key:
                    full_response = "⚠️ OPENAI_API_KEY not set. Set it and restart."
                    st.error(full_response)
                else:
                    try:
                        from openai import OpenAI
                        oa_client = OpenAI(api_key=openai_key)
                        oai_messages = []
                        if system_prompt:
                            oai_messages.append({"role": "system", "content": system_prompt})
                        oai_messages.extend(api_messages)

                        if st.session_state.use_tools:
                            # Agentic tool loop — LLM calls tools until it's ready to answer
                            for _round in range(6):
                                r = oa_client.chat.completions.create(
                                    model=model_id,
                                    messages=oai_messages,
                                    tools=build_openai_tools(),
                                    tool_choice="auto",
                                    max_tokens=max(st.session_state.max_tokens, 256),
                                )
                                msg = r.choices[0].message
                                if not msg.tool_calls:
                                    full_response = msg.content or ""
                                    break
                                oai_messages.append(msg)
                                for tc in msg.tool_calls:
                                    args = json.loads(tc.function.arguments)
                                    if tc.function.name == "calculator":
                                        expr = args.get("expression", "")
                                        result = calculate(expr)
                                        tci = {"name": "calculator", "query": expr,
                                               "calc_result": result, "web_results": []}
                                        tool_result_content = result
                                    else:
                                        query = args.get("query", "")
                                        with st.spinner(f'🔍 Searching: "{query}"...'):
                                            wr = web_search(query)
                                        tci = {"name": tc.function.name, "query": query,
                                               "calc_result": "", "web_results": wr}
                                        tool_result_content = format_search_results(wr)
                                    tool_calls_list.append(tci)
                                    oai_messages.append({
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": tool_result_content,
                                    })

                        if not full_response:
                            # Final streaming answer (tools done or tools disabled)
                            def response_generator():
                                stream = oa_client.chat.completions.create(
                                    model=model_id,
                                    messages=oai_messages,
                                    max_tokens=st.session_state.max_tokens,
                                    stream=True,
                                )
                                for chunk in stream:
                                    delta = chunk.choices[0].delta.content
                                    if delta:
                                        yield delta
                            full_response = st.write_stream(response_generator())
                        else:
                            st.markdown(full_response)
                    except Exception as exc:
                        full_response = f"⚠️ API error: {exc}"
                        st.error(full_response)

            else:  # Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    full_response = "⚠️ ANTHROPIC_API_KEY not set. Set it and restart."
                    st.error(full_response)
                else:
                    try:
                        client = Anthropic(api_key=api_key)
                        anth_messages = list(api_messages)

                        if st.session_state.use_tools:
                            # Agentic tool loop — LLM calls tools until it's ready to answer
                            for _round in range(6):
                                kwargs_loop: dict = dict(
                                    model=model_id,
                                    messages=anth_messages,
                                    tools=build_anthropic_tools(),
                                    max_tokens=max(st.session_state.max_tokens, 256),
                                )
                                if system_prompt:
                                    kwargs_loop["system"] = system_prompt
                                r = client.messages.create(**kwargs_loop)
                                if r.stop_reason != "tool_use":
                                    full_response = next(
                                        (b.text for b in r.content if hasattr(b, "text")), ""
                                    )
                                    break
                                tool_results = []
                                anth_messages = anth_messages + [
                                    {"role": "assistant", "content": r.content}
                                ]
                                for tu in (b for b in r.content if b.type == "tool_use"):
                                    if tu.name == "calculator":
                                        expr = tu.input.get("expression", "")
                                        result = calculate(expr)
                                        tci = {"name": "calculator", "query": expr,
                                               "calc_result": result, "web_results": []}
                                        tool_result_content = result
                                    else:
                                        query = tu.input.get("query", "")
                                        with st.spinner(f'🔍 Searching: "{query}"...'):
                                            wr = web_search(query)
                                        tci = {"name": tu.name, "query": query,
                                               "calc_result": "", "web_results": wr}
                                        tool_result_content = format_search_results(wr)
                                    tool_calls_list.append(tci)
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": tu.id,
                                        "content": tool_result_content,
                                    })
                                anth_messages = anth_messages + [
                                    {"role": "user", "content": tool_results}
                                ]

                        if not full_response:
                            # Final streaming answer (tools done or tools disabled)
                            def response_generator():
                                kwargs2: dict = dict(
                                    model=model_id,
                                    messages=anth_messages,
                                    max_tokens=st.session_state.max_tokens,
                                )
                                if system_prompt:
                                    kwargs2["system"] = system_prompt
                                with client.messages.stream(**kwargs2) as stream:
                                    for text in stream.text_stream:
                                        yield text
                            full_response = st.write_stream(response_generator())
                        else:
                            st.markdown(full_response)
                    except Exception as exc:
                        full_response = f"⚠️ API error: {exc}"
                        st.error(full_response)

    # Build trace with actual results after the interaction
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
