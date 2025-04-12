from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the credentials from environment variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Connect to Qdrant
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Print the available collections
print(client.get_collections())
