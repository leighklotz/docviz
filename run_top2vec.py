#!/usr/bin/env python3
import os
import sys
import glob
import umap
import matplotlib
import matplotlib.pyplot as plt
from top2vec import Top2Vec

# Use a non-interactive backend to prevent GUI issues in command-line environments
matplotlib.use('Agg')

def load_documents_from_directory(directory_path):
    """
    Reads all text files from a directory and returns a list of strings.
    """
    filenames = glob.glob(os.path.join(directory_path, '*'))
    documents = [process_file(file_path) for file_path in filenames]
    print(f"Successfully loaded {len(documents)} documents.")
    return documents

def process_file(file_path):
    print(f"{file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
    except Exception as e:
        import pdb;pdb.set_trace()
        raise Exception(f"Error reading file {file_path}", exc_info=True)

def create_and_save_umap_plot(model, output_filename='topic_umap_plot.png'):
    """
    Generates UMAP coordinates from the model and saves a scatter plot to a file.
    """
    print(f"\nGenerating UMAP visualization...")
    
    # 1. Get the original high-dimensional document vectors
    doc_vectors = model.document_vectors
    
    # 2. Re-run UMAP to get 2D coordinates for plotting
    # These parameters are a good starting point but can be tuned
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', verbose=True, random_state=42)
    umap_embeddings = umap_model.fit_transform(doc_vectors)
    
    # 3. Get the topic IDs for each document to color the plot
    # model.doc_top maps each document to its assigned topic ID
    doc_topics = model.doc_top
    
    # 4. Create the plot
    plt.figure(figsize=(12, 10))
    # Use topic IDs as colors for a simple visualization
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=doc_topics, cmap='Spectral', s=10, alpha=0.7)
    
    # Add a legend
    legend = plt.legend(*scatter.legend_elements(), title="Topics", loc="lower left", bbox_to_anchor=(1, 0))
    plt.gca().add_artist(legend)
    
    plt.title('UMAP Projection of Document Embeddings with Topics')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Save the figure instead of showing it
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"UMAP plot saved to '{output_filename}'")


# --- Main execution ---
if __name__ == "__main__":
    documents_list = load_documents_from_directory(sys.argv[1])

    if documents_list:
        print("\nStarting Top2Vec model training (this may take a while)...")
        # Train the model with default settings
        model = Top2Vec(documents=documents_list, speed="learn", workers=os.cpu_count())

        # Print text results
        num_topics = model.get_num_topics()
        print(f"\nTop2Vec successfully found {num_topics} topics.")
        topic_words, _, topic_nums_all = model.get_topics()
        print("\n--- Detected Topics and Top Words ---")
        for i, topic_num in enumerate(topic_nums_all):
            print(f"Topic #{topic_num}: Words: {', '.join(topic_words[i][:5])}")
            print("-" * 30)

        # Create and save the UMAP visualization
        create_and_save_umap_plot(model, output_filename='topic_umap_plot.png')
