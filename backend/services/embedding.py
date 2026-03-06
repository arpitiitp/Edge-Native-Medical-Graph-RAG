from sentence_transformers import SentenceTransformer

# Initialize the embedding model. This will download the model weights (~80MB) on first run.
# all-MiniLM-L6-v2 maps text to a 384 dimensional dense vector space.
print("Loading Embedding Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding Model Loaded.")

def generate_embedding(text: str) -> list[float]:
    """Generates a dense vector embedding for the given text."""
    # Ensure text is not empty and encode
    if not text.strip():
        return [0.0] * 384
    return model.encode(text).tolist()
