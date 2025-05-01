def compare_embeddings(text="This is a sample text to embed"):
    """Compare embeddings from different models"""
    # Define models to compare
    models = [
        "all-MiniLM-L6-v2",  # Small, fast model
        "all-mpnet-base-v2",  # Higher quality model
        # Add other models as needed
    ]
    
    results = {}
    for model_name in models:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        embeddings = embedding_model.embed_query(text)
        
        results[model_name] = {
            "dimensions": len(embeddings),
            "sample": embeddings[:3],  # First few values
            "min": min(embeddings),
            "max": max(embeddings),
            "avg": sum(embeddings)/len(embeddings)
        }
    
    # Print comparison
    for model, data in results.items():
        print(f"\n{model}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    return results