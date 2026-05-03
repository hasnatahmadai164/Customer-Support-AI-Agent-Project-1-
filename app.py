import streamlit as st
import sqlite3
import pandas as pd
from agent import run_agent
from tools import DB_PATH


st.set_page_config(
    page_title="Customer Support AI Agent",
    page_icon="🛒",
    layout="wide"
)

# CUSTOM CSS
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

# SESSION STATE — PERSISTENT MEMORY ACROSS RE-RUNS

if "messages" not in st.session_state:
    st.session_state.messages = []        

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    #

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
        return pd.DataFrame()   
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

    if not st.session_state.messages:
        st.info(
            "Hi! I'm Customer Support Agent"
            "I can help with returns, shipping, orders, payments and account issues. "
            "How can I help you today?"
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Type your message here..."):

        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("ShopBot is thinking..."):
            response = run_agent(
                user_message=user_input,
                chat_history=st.session_state.chat_history
            )

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

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



with col_dashboard:

    st.markdown("###  Support Ticket Dashboard")
    st.caption("Live view of all tickets created by the agent")

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

        for _, ticket in tickets_df.iterrows():

            # Pick an icon based on ticket status
            if ticket["escalated"] == 1:
                icon = "🔴"
            elif ticket["status"] == "open":
                icon = "🟡"
            else:
                icon = "🟢"

            label = f"{icon} #{ticket['id']} — {ticket['issue_category']} ({ticket['priority']})"
            with st.expander(label):
                st.write(f"**Customer:** {ticket['customer_name']}")
                st.write(f"**Email:** {ticket['customer_email']}")
                st.write(f"**Status:** {ticket['status']}")
                st.write(f"**Created:** {ticket['created_at']}")
                st.write("**Issue Description:**")

                # key must be unique per widget — we use the ticket ID
                st.text_area(
                    label="",
                    value=ticket["issue_description"],
                    height=90,
                    disabled=True,
                    key=f"desc_{ticket['id']}"
                )

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
