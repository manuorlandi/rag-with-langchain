from src.models.embeddings import get_embeddings

def inspect_embeddings(text="This is a sample text to embed"):
    """Inspect embeddings for a given text"""
    # Get the embedding model
    embedding_model = get_embeddings()
    
    # Generate embeddings for the text
    embeddings = embedding_model.embed_query(text)
    
    # Basic information
    print(f"Embedding dimensions: {len(embeddings)}")
    print(f"First 5 values: {embeddings[:5]}")
    print(f"Min value: {min(embeddings)}")
    print(f"Max value: {max(embeddings)}")
    print(f"Average value: {sum(embeddings)/len(embeddings)}")
    
    # Visualize (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(embeddings)
        plt.title("Embedding Vector Visualization")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.savefig("embedding_visualization.png")
        print("Visualization saved to embedding_visualization.png")
    except ImportError:
        print("Matplotlib not available for visualization")
    
    return embeddings

# Example usage
if __name__ == "__main__":
    inspect_embeddings()