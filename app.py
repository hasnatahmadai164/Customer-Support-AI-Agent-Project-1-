import streamlit as st
import sqlite3
import pandas as pd
from agent import run_agent
from tools import DB_PATH

# PAGE CONFIG
# Sets browser tab title, icon, and page layout

st.set_page_config(
    page_title="Customer Support AI Agent",
    page_icon="🛒",
    layout="wide"
)

# CUSTOM CSS
#
# Streamlit's default UI is functional but basic.
# We inject a small CSS block to improve the visual quality.
# unsafe_allow_html=True is required to render raw HTML/CSS.
# ============================================================

st.markdown("""
<style>
    /* Light background for the whole app */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Rounded chat bubbles */
    .stChatMessage {
        border-radius: 14px;
        margin-bottom: 6px;
    }

    /* Gradient header banner */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        padding: 22px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        text-align: center;
    }

    /* Green online status pill */
    .status-pill {
        background-color: #34a853;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE — PERSISTENT MEMORY ACROSS RE-RUNS
#
# Streamlit re-runs the full script on every user action.
# Without session state, all variables would reset each time.
# st.session_state is a dictionary that survives re-runs.
#
# We store two lists:
#   messages      → for displaying in the chat UI (what the user sees)
#   chat_history  → for passing to the agent (same data, same format)
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []        # UI display history

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # Agent input history


# ============================================================
# HELPER: Load tickets from SQLite for the dashboard
# ============================================================

def load_tickets() -> pd.DataFrame:
    """
    Reads all tickets from SQLite database and returns as a DataFrame.
    pandas DataFrame makes it easy for Streamlit to display as a table.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM tickets ORDER BY created_at DESC",
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()   # empty DataFrame if table doesn't exist yet


# ============================================================
# PAGE LAYOUT — TWO COLUMNS
#
# st.columns([2, 1]) creates two side-by-side sections.
# [2, 1] ratio: left column is 2x wider than right column.
# ============================================================

col_chat, col_dashboard = st.columns([2, 1])

# LEFT COLUMN: CHAT INTERFACE

with col_chat:

    # Header banner
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-size:24px;"
        <p style="margin:6px 0 8px 0; opacity:0.85;">Powered by AI — here to help 24/7</p>
        <span class="status-pill">● Online</span>
    </div>
    """, unsafe_allow_html=True)

    #Welcome message on first load (before any messages exist)
    if not st.session_state.messages:
        st.info(
            "Hi! I'm Customer Support Agent"
            "I can help with returns, shipping, orders, payments and account issues. "
            "How can I help you today?"
        )

    #Display all previous messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Chat input box (fixed at the bottom of the page)
    if user_input := st.chat_input("Type your message here..."):

        # 1. Show the user's message immediately in the chat
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Save it to session state so it persists on next re-run
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # 3. Show a spinner while the agent processes (ReAct loop runs here)
        with st.spinner("ShopBot is thinking..."):
            response = run_agent(
                user_message=user_input,
                chat_history=st.session_state.chat_history
            )

        # 4. Display the agent's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # 5. Save the agent's response to display history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # 6. Update chat_history 
        # It gives the agent memory of the full conversation
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

    # Clear conversation button
    if st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()



# RIGHT COLUMN: TICKET DASHBOARD
with col_dashboard:

    st.markdown("###  Support Ticket Dashboard")
    st.caption("Live view of all tickets created by the agent")

    # Manual refresh button — triggers a re-run to reload DB data
    if st.button("Refresh", type="secondary"):
        st.rerun()

    tickets_df = load_tickets()

    if tickets_df.empty:
        st.info("No tickets yet. Start chatting to create one!")

    else:
        # Summary metrics at the top
        total = len(tickets_df)
        open_count = len(tickets_df[tickets_df["status"] == "open"])
        escalated_count = len(tickets_df[tickets_df["escalated"] == 1])

        # st.metric() shows a large number with a label underneath
        m1, m2, m3 = st.columns(3)
        m1.metric("Total", total)
        m2.metric("Open", open_count)
        m3.metric("🔴 Escalated", escalated_count)

        st.divider()

        #Individual ticket cards 
        # Loop through each ticket and show it as a collapsible expander
        for _, ticket in tickets_df.iterrows():

            # Pick an icon based on ticket status
            if ticket["escalated"] == 1:
                icon = "🔴"
            elif ticket["status"] == "open":
                icon = "🟡"
            else:
                icon = "🟢"

            # st.expander() creates a collapsible section the user can click
            label = f"{icon} #{ticket['id']} — {ticket['issue_category']} ({ticket['priority']})"
            with st.expander(label):
                st.write(f"**Customer:** {ticket['customer_name']}")
                st.write(f"**Email:** {ticket['customer_email']}")
                st.write(f"**Status:** {ticket['status']}")
                st.write(f"**Created:** {ticket['created_at']}")
                st.write("**Issue Description:**")

                # st.text_area with disabled=True shows scrollable read-only text
                # key must be unique per widget — we use the ticket ID
                st.text_area(
                    label="",
                    value=ticket["issue_description"],
                    height=90,
                    disabled=True,
                    key=f"desc_{ticket['id']}"
                )

    # --- Sample questions for testing the agent ---
    st.divider()
    st.markdown("### 💡 Try These Prompts")
    st.markdown("""
- *"What is your return policy?"*
- *"How long does standard shipping take?"*
- *"Can I cancel my order?"*
- *"I want to talk to a human agent"*
- *"My name is Ali, email ali@test.com — I received a damaged item"*
- *"My package shows delivered but I never got it"*
    """)
