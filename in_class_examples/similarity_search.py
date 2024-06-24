from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema.document import Document
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize the embeddings class
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Embed a single query
query = "Hello, world!"
vector = embeddings.embed_query(query)
print(vector[:5])

# Embed multiple documents at once
documents = ["Alice works in finance", "Bob is a database administrator", "Carl manages Bob and Alice"]

# Convert the list of strings to a list of Document objects
document_objects = [Document(page_content=doc) for doc in documents]

# Embed the Document objects
vectors = embeddings.embed_documents([doc.page_content for doc in document_objects])
print(len(vectors), len(vectors[0]))

# Initialize the FAISS database with the Document objects and embeddings
db = FAISS.from_documents(document_objects, embeddings)

query = "Tell me about Alice"
docs = db.similarity_search(query)

# Perform a similarity search with scores
docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores)
