from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Document
from constants import *
import pandas as pd
from llama_index.core import Settings

data = pd.read_csv("eproducts.csv")

descriptions = data["Description"].tolist()
actual_categories = data["Category"].tolist()

# Print some sample data to verify
# print("Sample data:")
for desc, cat in zip(descriptions[:5], actual_categories[:5]):
    print(f"Description: {desc}")
    print(f"Category: {cat}")
    print("-" * 50)

# Initialize the Gemini model
Settings.llm = Gemini(model="models/gemini-pro", api_key=API_KEY)

gemini_embedding = GeminiEmbedding(model="models/gemini-pro", api_key=API_KEY)

Settings.embed_model = gemini_embedding

# Define a custom query prompt to group products into categories
query_prompt = PromptTemplate(
    template="""Group the following product descriptions into these two main categories: Clothing and Accessories.
    For each category, list the products that belong to it.
    
    Format your response exactly like this:
    Clothing:
    - [list products that are clothing items]
    
    Accessories:
    - [list products that are accessories]
    {query_str}"""
)


# Create an index
# documents = [{"text": desc} for desc in descriptions]

# Create proper Document objects with category metadata
documents = []
for i, desc in enumerate(descriptions):
    doc = Document(
        text=desc,  # The main text content
        id_=f"doc_{i}",  # Optional: unique identifier
        metadata={  # Optional: any metadata you want to attach
            "index": i,
            "source": "eproducts.csv",
        },
    )
    documents.append(doc)
index = VectorStoreIndex.from_documents(documents)

# Query to group descriptions
query_engine = index.as_query_engine(query_prompt=query_prompt)
response = query_engine.query(
    "Categorize these products into Clothing and Accessories only."
)
# print(response)
# print(dir(response))

# Example: Parsing response for assigning categories
grouped_data = (
    response.get_formatted_sources()
)  # Adjust based on API's response structure
print(grouped_data)
