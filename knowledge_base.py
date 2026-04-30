import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# SAMPLE KNOWLEDGE BASE DOCUMENTS
documents = [

    # RETURNS AND REFUNDS 
    Document(
        page_content=(
            "Our return policy allows customers to return any item within 30 days of delivery. "
            "Items must be unused, in original packaging, and accompanied by the order receipt. "
            "To initiate a return, customers should visit the Returns Portal on our website or "
            "contact support. Refunds are processed within 5-7 business days after we receive "
            "the returned item. Items marked as 'Final Sale' are not eligible for returns."
        ),
        metadata={"source": "returns_policy", "category": "returns"}
    ),

    Document(
        page_content=(
            "Damaged or defective items can be reported within 48 hours of delivery. "
            "Customers should take a photo of the damage and submit it through the support portal. "
            "We will arrange a free return pickup and send a replacement within 3-5 business days. "
            "No restocking fee is charged for damaged items."
        ),
        metadata={"source": "damaged_items_policy", "category": "returns"}
    ),

    # SHIPPING
    Document(
        page_content=(
            "Standard shipping takes 5-7 business days and is free on orders over $50. "
            "Express shipping (2-3 business days) costs $12.99. "
            "Overnight shipping is available for $24.99 and must be ordered before 12pm EST. "
            "International shipping is available to 45 countries and typically takes 10-20 business days. "
            "Shipping costs for international orders depend on destination and weight."
        ),
        metadata={"source": "shipping_policy", "category": "shipping"}
    ),

    Document(
        page_content=(
            "Once an order is shipped, customers receive a tracking number via email. "
            "Tracking information may take up to 24 hours to update after the shipment is picked up. "
            "If a package shows as delivered but has not arrived, customers should: "
            "1) Check with neighbors, 2) Check all entry points, 3) Wait 24 hours, "
            "then contact support if still missing. We will file a carrier claim on your behalf."
        ),
        metadata={"source": "tracking_policy", "category": "shipping"}
    ),

    # ORDERS
    Document(
        page_content=(
            "Orders can be cancelled within 1 hour of placement at no charge. "
            "After 1 hour, if the order has not yet been shipped, a $5 cancellation fee applies. "
            "Once an order is shipped, it cannot be cancelled — the customer must initiate a return. "
            "To cancel an order, visit 'My Orders' in your account dashboard or contact support."
        ),
        metadata={"source": "order_cancellation_policy", "category": "orders"}
    ),

    Document(
        page_content=(
            "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. "
            "All transactions are encrypted with SSL. We do not store full card details. "
            "If a payment fails, please check your billing address matches your card details. "
            "For PayPal issues, contact PayPal support directly. "
            "Buy Now Pay Later is available via Klarna for orders over $100."
        ),
        metadata={"source": "payment_policy", "category": "orders"}
    ),

    # ACCOUNT
    Document(
        page_content=(
            "To reset your password, click 'Forgot Password' on the login page. "
            "A reset link will be emailed to your registered email address. "
            "The link expires in 15 minutes. If you do not receive the email, check your spam folder. "
            "For account lockouts after 5 failed attempts, wait 30 minutes or contact support."
        ),
        metadata={"source": "account_policy", "category": "account"}
    ),

    Document(
        page_content=(
            "Loyalty points are earned at a rate of 1 point per $1 spent. "
            "Points can be redeemed for discounts: 100 points = $1 off. "
            "Points expire after 12 months of account inactivity. "
            "Bonus points are awarded during promotional events and on birthdays. "
            "Points cannot be transferred between accounts or redeemed for cash."
        ),
        metadata={"source": "loyalty_program", "category": "account"}
    ),
]


def create_pinecone_index(pc: Pinecone) -> object:
    """
    Creates a Pinecone index if it does not already exist, then returns it.

    PINECONE v8 CHANGE:
    We now use pc.has_index(name) to check existence — cleaner than the
    old approach of listing all indexes and checking manually.

    Args:
        pc: An initialized Pinecone client object.

    Returns:
        A Pinecone Index object ready to use.
    """

    index_name = os.getenv("PINECONE_INDEX_NAME")

    # pc.has_index() — new in Pinecone v8, cleaner existence check
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index: '{index_name}'...")

        pc.create_index(
            name=index_name,

            # 1536 dimensions = output size of text-embedding-3-small
            # This number MUST match the embedding model you use
            dimension=1536,

            metric="cosine",

            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD"),
                region=os.getenv("PINECONE_REGION")
            )
        )

        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        print(f"Index '{index_name}' is ready.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

        return pc.Index(index_name)


def ingest_documents():
    """
    Converts documents to embeddings and uploads them to Pinecone.

    FLOW:
    1. Initialize Pinecone client
    2. Create index if needed → get Index object
    3. Initialize OpenAI embedding model
    4. Initialize PineconeVectorStore with the Index object (v8 pattern)
    5. Add documents → LangChain embeds + uploads automatically
    """

    print("Starting document ingestion...")

    # Step 1: Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Step 2: Create index and get the Index object
    index = create_pinecone_index(pc)

    # Step 3: Initialize the embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Step 4: Initialize PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # Step 5: Add documents to Pinecone
    vectorstore.add_documents(documents=documents)

    print(f"Successfully uploaded {len(documents)} documents to Pinecone!")
    print("Knowledge base is ready. Now run: streamlit run app.py")

    return vectorstore


if __name__ == "__main__":
    ingest_documents()
