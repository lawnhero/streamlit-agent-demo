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
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = {
    "Haiku 4.5 (fast)": "claude-haiku-4-5-20251001",
    "Sonnet 4.5 (smart)": "claude-sonnet-4-5",
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
    ("What did I just ask you?",           "🧠", "Tests Memory — try OFF then ON"),
    ("What's 847 × 293?",                  "🔢", "Tests Tools + Workflow"),
    ("What's the refund policy?",          "📚", "Tests RAG — try OFF then ON"),
    ("What's in the news today?",          "📰", "Tests Tools — try OFF then ON"),
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
    "model_label": "Haiku 4.5 (fast)",
    "max_tokens": 1000,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Core helpers ──────────────────────────────────────────────────────────────

def rag_retrieve(query: str) -> list[dict]:
    q = query.lower()
    return [d for d in KNOWLEDGE_BASE if any(kw in q for kw in d["keywords"])]


def build_system_prompt(history: list[dict], rag_docs: list[dict]) -> str:
    s = st.session_state
    parts = [
        "You are 'Little L', a helpful assistant in a business school classroom "
        "demo about AI agent architecture."
    ]

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
            "\n\n## Tools Available\n"
            "- **web_search(query)**: Search the internet for current info. "
            "Say 'Searching: [query]...' before using it.\n"
            "- **calculator(expression)**: Evaluate math precisely. "
            "Say 'Calculating: [expr] = [result]' when using it."
        )

    if s.use_workflow:
        parts.append(
            "\n\n## Routing Rule (Hardcoded Workflow)\n"
            "If the question is PURELY arithmetic, respond with exactly: "
            "'Routing to calculator: [answer]' — nothing else."
        )

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


def build_trace(user_msg: str, rag_docs: list[dict]) -> list[tuple[str, str]]:
    s = st.session_state
    msg = user_msg.lower()
    steps: list[tuple[str, str]] = []

    is_math = bool(
        re.search(r"\b\d[\d\s]*[\+\-×\*\/x÷]\s*\d", msg)
        or any(w in msg for w in ["times", "multiply", "divided", "plus", "minus"])
    )
    is_news = any(w in msg for w in ["news", "today", "recent", "latest", "current events"])
    is_doc_q = any(kw in msg for d in KNOWLEDGE_BASE for kw in d["keywords"])
    is_mem_q = bool(re.search(r"what did i (say|ask)|previous|before|last message|remember", msg))

    steps.append(("⚪", "User message received by agent"))

    if s.use_workflow and is_math:
        steps.append(("🟡", "⚡ WORKFLOW: math query detected"))
        steps.append(("🟠", "🔧 TOOL: routing to calculator — bypassing LLM entirely"))
        steps.append(("🟡", "⚡ WORKFLOW: result returned (0 LLM reasoning tokens used)"))
        return steps

    n_hist = len(s.messages)
    if s.use_memory:
        if n_hist > 0:
            n = min(n_hist, 6)
            steps.append(("🔵", f"🧠 MEMORY: fetching last {n} message(s) from conversation store"))
            steps.append(("🔵", f"🧠 MEMORY: {n} message(s) appended to system prompt"))
        else:
            steps.append(("🔵", "🧠 MEMORY: enabled — no prior messages yet"))
    elif is_mem_q:
        steps.append(("❌", "No memory — agent cannot recall prior messages"))

    if s.use_rag:
        steps.append(("🟣", "📚 RAG: embedding query for similarity search..."))
        if rag_docs:
            titles = ", ".join(d["title"] for d in rag_docs)
            steps.append(("🟣", f"📚 RAG: {len(rag_docs)} chunk(s) retrieved — {titles}"))
            steps.append(("🟣", "📚 RAG: chunks injected into system prompt"))
        else:
            steps.append(("🟣", "📚 RAG: no relevant documents found in knowledge base"))
    elif is_doc_q:
        steps.append(("❌", "No RAG — cannot retrieve from company knowledge base"))

    if s.use_tools:
        if is_news:
            steps.append(("🟠", f'🔧 TOOLS: LLM emits tool call → web_search("{user_msg[:50]}...")'))
            steps.append(("🟠", "🔧 TOOLS: results fetched → appended to context"))
        if is_math:
            steps.append(("🟠", "🔧 TOOLS: LLM emits tool call → calculator(...)"))
    elif is_news:
        steps.append(("❌", "No tools — cannot search the web for live information"))

    steps.append(("⚪", "→ Assembled prompt sent to LLM"))
    steps.append(("⚪", "← LLM streams tokens back to user"))
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
        "Max tokens to generate", min_value=1, max_value=4096,
        value=st.session_state.max_tokens, step=1,
        help="Caps the length of the response. Lower = shorter answers and faster responses.",
    )
    st.caption(f"~{st.session_state.max_tokens // 4 * 3} words max")

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

    st.toggle("⚡ Workflow Logic", key="use_workflow")
    st.caption(
        "✅ Math queries bypass LLM → go direct to calculator" if st.session_state.use_workflow
        else "🚫 All requests flow through LLM"
    )

    if st.button("↺ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.session_state.last_trace = []
        st.session_state.last_system_prompt = ""
        st.session_state.last_rag_docs = []
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
if active:
    st.success(f"**Active components:** {' · '.join(active)}")
else:
    st.info("**Active:** Bare LLM — no components enabled. Toggle something in the sidebar!")

col_chat, col_inspector = st.columns([3, 2], gap="large")

# ── Inspector column ──────────────────────────────────────────────────────────
with col_inspector:
    st.markdown("### 🔍 Under the Hood")
    t_trace, t_prompt, t_rag = st.tabs(["⚡ Process Trace", "📋 System Prompt", "📚 RAG Docs"])

    with t_trace:
        if st.session_state.last_trace:
            for emoji, text in st.session_state.last_trace:
                st.markdown(f"{emoji} &nbsp; {text}")
        else:
            st.caption("Step-by-step trace will appear here after your first message.")
            st.markdown(
                "**What you'll see:**\n"
                "- Which components activated\n"
                "- Order of operations (memory fetch → RAG retrieval → tool calls)\n"
                "- Whether the LLM was bypassed by workflow logic"
            )

    with t_prompt:
        # Live preview — updates as toggles change, before any message is sent
        live_preview = build_system_prompt([], [])
        st.caption(f"**Live preview** — updates as you toggle components (~{len(live_preview)//4} tokens)")
        st.code(live_preview, language="text")

        if (
            st.session_state.last_system_prompt
            and st.session_state.last_system_prompt != live_preview
        ):
            st.divider()
            full_toks = len(st.session_state.last_system_prompt) // 4
            st.caption(f"**Last prompt actually sent** (includes history + retrieved docs) — ~{full_toks} tokens")
            st.code(st.session_state.last_system_prompt, language="text")

    with t_rag:
        if st.session_state.use_rag:
            if st.session_state.last_rag_docs:
                st.success(f"Last query retrieved **{len(st.session_state.last_rag_docs)}** chunk(s):")
                for doc in st.session_state.last_rag_docs:
                    with st.expander(f"📄 {doc['title']}", expanded=True):
                        st.write(doc["content"])
                        st.caption(f"Matched keywords: {', '.join(doc['keywords'])}")
            else:
                st.warning("RAG is ON but no documents matched the last query.")
            st.divider()
            st.caption("**Full knowledge base** (4 documents):")
            for doc in KNOWLEDGE_BASE:
                with st.expander(f"📄 {doc['title']}"):
                    st.write(doc["content"])
                    st.caption(f"Keywords: {', '.join(doc['keywords'])}")
        else:
            st.caption("Enable RAG to see document retrieval in action.")
            st.markdown(
                "**When enabled**, this panel shows:\n"
                "- Which chunks were retrieved for your query\n"
                "- The full knowledge base being searched\n"
                "- Which keywords triggered the match"
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
    system_prompt = build_system_prompt(history, rag_docs)
    trace = build_trace(prompt, rag_docs)

    # Persist for inspector (visible after rerun)
    st.session_state.last_trace = trace
    st.session_state.last_system_prompt = system_prompt
    st.session_state.last_rag_docs = rag_docs

    # Build API messages
    if st.session_state.use_memory:
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
        ]
    else:
        api_messages = []
    api_messages.append({"role": "user", "content": prompt})

    input_tokens = len(system_prompt) // 4 + sum(len(m["content"]) // 4 for m in api_messages)

    # Stream into the placeholder inside col_chat
    with stream_slot.container():
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                full_response = "⚠️ ANTHROPIC_API_KEY not set. Set it and restart."
                st.error(full_response)
            else:
                try:
                    client = Anthropic(api_key=api_key)
                    model_id = MODELS[st.session_state.model_label]

                    def response_generator():
                        with client.messages.stream(
                            model=model_id,
                            max_tokens=st.session_state.max_tokens,
                            system=system_prompt,
                            messages=api_messages,
                        ) as stream:
                            for text in stream.text_stream:
                                yield text

                    full_response = st.write_stream(response_generator())
                except Exception as exc:
                    full_response = f"⚠️ API error: {exc}"
                    st.error(full_response)

    # Persist messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.token_count += input_tokens + len(full_response) // 4

    st.rerun()
