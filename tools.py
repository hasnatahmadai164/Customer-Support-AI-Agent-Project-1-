import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

DB_PATH = "support_tickets.db"


def init_database():
    """
    Creates SQLite database and tickets table if they do not exist yet.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            customer_email TEXT,
            issue_category TEXT,
            issue_description TEXT,
            status TEXT DEFAULT 'open',
            priority TEXT DEFAULT 'normal',
            created_at TEXT,
            escalated INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


init_database()


def get_retriever():
    """
    Creates and returns a LangChain retriever backed by Pinecone.

    A retriever converts a query string into a vector, searches
    Pinecone for the top-k most similar document vectors, and
    returns those documents for the agent to read.

    Returns:
        A LangChain VectorStoreRetriever that fetches top 3 results.
    """

    # Initialize embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize Pinecone client 
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Get the Index object
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    # Initialize PineconeVectorStore with the Index object 
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # as_retriever() wraps the vectorstore as a retriever
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# TOOL 1: SEARCH KNOWLEDGE BASE (RAG TOOL)

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the e-commerce support knowledge base to find answers
    to customer questions about returns, shipping, orders, payments,
    and account issues. Always use this tool FIRST before answering
    any policy-related question.

    Args:
        query: The customer's question or topic to search for.

    Returns:
        Relevant policy information from the knowledge base.
    """

    try:
        retriever = get_retriever()

        if not docs:
            return "No relevant information found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

@tool
def create_support_ticket(
    customer_name: str,
    customer_email: str,
    issue_category: str,
    issue_description: str,
    priority: str = "normal"
) -> str:
    """
    Create a support ticket in the database when a customer has an
    issue that requires follow-up and cannot be resolved immediately.
    Use this when the customer reports a missing package, damaged item,
    failed payment, or refund request.
    Always ask for the customer's name and email before calling this tool.

    Args:
        customer_name: Full name of the customer.
        customer_email: Email address of the customer.
        issue_category: Category such as 'shipping', 'returns', 'payment', or 'account'.
        issue_description: Detailed description of the customer's problem.
        priority: One of 'low', 'normal', or 'high'. Defaults to 'normal'.

    Returns:
        Confirmation message with the new ticket ID.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO tickets 
            (customer_name, customer_email, issue_category, issue_description, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (customer_name, customer_email, issue_category, issue_description, priority, created_at))

        conn.commit()
        ticket_id = cursor.lastrowid
        conn.close()

        return (
            f"Support ticket created successfully!\n"
            f"Ticket ID: #{ticket_id}\n"
            f"Customer: {customer_name} ({customer_email})\n"
            f"Category: {issue_category}\n"
            f"Priority: {priority}\n"
            f"Status: Open\n"
            f"Our team will follow up within 24 hours."
        )

    except Exception as e:
        return f"Error creating support ticket: {str(e)}"


@tool
def escalate_to_human(
    customer_name: str,
    customer_email: str,
    reason: str,
    conversation_summary: str
) -> str:
    """
    Escalate the conversation to a human support agent when:
    - The customer is very frustrated, angry, or threatening legal action
    - The issue involves fraud, security, or account compromise
    - You cannot find a satisfactory answer after searching the knowledge base
    - The customer explicitly requests to speak with a human agent
    Always ask for name and email before calling this tool.

    Args:
        customer_name: Full name of the customer.
        customer_email: Email address of the customer.
        reason: Brief explanation of why you are escalating.
        conversation_summary: Summary of the conversation so far,
                              so the human agent has full context.

    Returns:
        Confirmation that escalation has been registered.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO tickets 
            (customer_name, customer_email, issue_category, issue_description,
             priority, status, created_at, escalated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            customer_name,
            customer_email,
            "escalation",
            f"ESCALATION REASON: {reason}\n\nCONVERSATION SUMMARY:\n{conversation_summary}",
            "high",
            "escalated",
            created_at,
            1  
        ))

        conn.commit()
        ticket_id = cursor.lastrowid
        conn.close()

        return (
            f"Escalation registered. Ticket #{ticket_id} is flagged as HIGH PRIORITY.\n"
            f"A human support agent will contact {customer_name} at {customer_email} "
            f"within 2 hours.\n"
            f"Escalation reason: {reason}"
        )

    except Exception as e:
        return f"Error registering escalation: {str(e)}"


all_tools = [
    search_knowledge_base,
    create_support_ticket,
    escalate_to_human
]
