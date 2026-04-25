import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from tools import all_tools

load_dotenv()

# SYSTEM PROMPT
SYSTEM_PROMPT = """You are ShopBot, a friendly and professional customer support agent 
for ShopEasy — an e-commerce platform. Your job is to help customers resolve issues 
quickly and accurately.

## YOUR TOOLS:
You have access to 3 tools. Use them wisely:

1. **search_knowledge_base**
   Search our internal knowledge base for policy information and answers.
   Use this FIRST for any question about returns, refunds, shipping, orders, 
   payments, or account issues. Never answer policy questions from memory alone.

2. **create_support_ticket**
   Create a ticket in our system when a customer has an issue needing follow-up.
   Use for: missing packages, damaged items, failed payments, refund requests.
   IMPORTANT: Always collect the customer's name and email BEFORE creating a ticket.

3. **escalate_to_human**
   Hand off to a human agent when the situation requires it.
   Escalate when: customer is very angry, issue involves fraud or security,
   you cannot resolve it after searching the knowledge base,
   or the customer explicitly asks for a human.
   IMPORTANT: Always collect the customer's name and email BEFORE escalating.

## YOUR RULES:
- ALWAYS search the knowledge base before answering policy questions.
- NEVER make up or guess policy information. Search first.
- Be warm, empathetic, and professional at all times.
- If a customer is frustrated, acknowledge their feelings before solving.
- If you create a ticket or escalate, confirm it to the customer with the ticket ID.
- Keep responses focused and easy to read. Use bullet points when helpful.
- If someone asks about something unrelated to shopping or your role, 
  politely redirect them.

## YOUR TONE:
- Friendly and human — not robotic
- Empathetic when customers are frustrated or upset
- Clear and direct — no jargon or corporate language
- Always end with an offer to help further
"""

# PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),   # conversation memory
    ("human", "{input}"),                                # current user message
    MessagesPlaceholder(variable_name="agent_scratchpad") # ReAct thinking space
])

# LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# CREATE THE REACT AGENT
agent = create_tool_calling_agent(
    llm=llm,
    tools=all_tools,
    prompt=prompt
)

# AGENT EXECUTOR
agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)


# run_agent() THE FUNCTION CALLED BY THE UI (app.py)
def run_agent(user_message: str, chat_history: list) -> str:
    """
    Run the support agent with the user's message and conversation history.

    Args:
        user_message: The latest message from the customer.
        chat_history: List of previous messages as dicts:
                      [{"role": "user", "content": "..."}, ...]

    Returns:
        The agent's response as a plain string.
    """

    langchain_history = []
    for message in chat_history:
        if message["role"] == "user":
            langchain_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_history.append(AIMessage(content=message["content"]))

    try:
        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": langchain_history
        })
        return response["output"]

    except Exception as e:
        return f"I'm sorry, I ran into an issue. Please try again. (Error: {str(e)})"
