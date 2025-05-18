"""
ClinicalBERT Semantic Clustering on MIMIC Discharge Summaries

Author: Tara K. J. et al.

Description:
This project applies KMeans clustering to ClinicalBERT embeddings generated
from discharge summary sentences sourced from the MIMIC-III dataset.
The output includes 3D visualizations (t-SNE and PCA), top keyword annotations,
and word cloud panels representing semantic clusters.

Key Steps:
- Load and preprocess discharge summaries
- Generate ClinicalBERT sentence embeddings
- Apply KMeans clustering to identify semantic clusters
- Apply LDA clustering to identify semantic clusters (# TODO)
- Visualize clusters using t-SNE and PCA
- Annotate clusters with most frequent words and export results
"""
import os
import re
from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
from wordcloud import WordCloud

# Global variables
FILE_PATH = "Data/multi_category.csv"
NUM_CLUSTERS = 4
OUTPUT_DIR = "Outputs"
SEED = 314159

os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download("stopwords")
nltk.download("wordnet")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def load_clinicalbert_model(model_name: str = "emilyalsentzer/Bio_ClinicalBERT") -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load the ClinicalBERT tokenizer and model from Hugging Face.

    Parameters:
        model_name (str): Name of the pretrained model to load.

    Returns:
        Tuple[AutoTokenizer, AutoModel]: Tokenizer and model instances.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def get_clinicalbert_embeddings(sentences: pd.Series, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """
    Generate ClinicalBERT embeddings for a list of sentences.

    Parameters:
        sentences (pd.Series): Input sentences.
        tokenizer (AutoTokenizer): Tokenizer for ClinicalBERT.
        model (AutoModel): ClinicalBERT model.

    Returns:
        np.ndarray: Embeddings array of shape (n_samples, embedding_dim).
    """
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)

def perform_clustering(embeddings: np.ndarray, num_clusters: int = NUM_CLUSTERS) -> np.ndarray:
    """
    Apply KMeans clustering to the sentence embeddings.

    Parameters:
        embeddings (np.ndarray): Sentence embeddings.
        num_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels for each sentence.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=SEED)
    return kmeans.fit_predict(embeddings)

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 3, perplexity: int = 30) -> np.ndarray:
    """
    Reduce dimensions of embeddings using t-SNE.

    Parameters:
        embeddings (np.ndarray): High-dimensional data.
        n_components (int): Number of dimensions for output.
        perplexity (int): Perplexity parameter for t-SNE.

    Returns:
        np.ndarray: Reduced embeddings of shape (n_samples, n_components).
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=SEED)
    return tsne.fit_transform(embeddings)

def visualize_clusters_3d(embeddings: np.ndarray, labels: np.ndarray, title: str, filename: str):
    """
    Generate and save a 3D scatter plot of clusters.

    Parameters:
        embeddings (np.ndarray): 3D-reduced embeddings.
        labels (np.ndarray): Cluster labels.
        title (str): Title for the plot.
        filename (str): Output filename for saving the figure.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
        c=labels, cmap="tab10", alpha=0.7
    )
    ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.show()

def visualize_pca(embeddings: np.ndarray, labels: np.ndarray) -> None:
    """
    Apply PCA to embeddings and visualize clusters in 3D.

    Parameters:
        embeddings (np.ndarray): High-dimensional embeddings.
        labels (np.ndarray): Cluster labels.
    """
    pca = PCA(n_components=3)
    pca_embeddings = pca.fit_transform(embeddings)
    visualize_clusters_3d(
        pca_embeddings,
        labels,
        title="3D PCA Visualization of ClinicalBERT Semantic Clusters",
        filename="pca_3d_clusters.png"
    )

def preprocess_text(text: str) -> list:
    """
    Clean text by removing stopwords and applying lemmatization.

    Parameters:
        text (str): Raw input text.

    Returns:
        list: List of processed and lemmatized tokens.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = re.findall(r"\b\w+\b", text.lower())
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]

def generate_wordclouds(data: pd.DataFrame, labels: np.ndarray, num_clusters: int = NUM_CLUSTERS) -> None:
    """
    Generate and save a panel of word clouds using top words per cluster.

    Parameters:
        data (pd.DataFrame): Input dataset.
        labels (np.ndarray): Cluster labels.
        num_clusters (int): Number of clusters.
    """
    fig, axes = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 5))
    for cluster in range(num_clusters):
        cluster_text = " ".join(data["sentence"][labels == cluster])
        processed_words = preprocess_text(cluster_text)
        most_common_words = dict(Counter(processed_words).most_common(10))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(most_common_words)
        axes[cluster].imshow(wordcloud, interpolation="bilinear")
        axes[cluster].axis("off")
        axes[cluster].set_title(f"Cluster {cluster}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds_top10_panel.png"))
    plt.show()

def save_labeled_dataset(data: pd.DataFrame, labels: np.ndarray, output_path: str = os.path.join(OUTPUT_DIR, "semantic_clustered_dataset.csv")) -> None:
    """
    Save the dataset with appended cluster labels to a CSV file.

    Parameters:
        data (pd.DataFrame): Input dataset.
        labels (np.ndarray): Cluster labels.
        output_path (str): File path for saving the CSV.
    """
    data["semantic_cluster"] = labels
    labeled_dataset = data[["sentence", "semantic_cluster"]]
    labeled_dataset.to_csv(output_path, index=False)

def main():

    data = load_data(FILE_PATH)
    tokenizer, model = load_clinicalbert_model()
    embeddings = get_clinicalbert_embeddings(data["sentence"], tokenizer, model)
    labels = perform_clustering(embeddings)

    tsne_embeddings = reduce_dimensions(embeddings)
    visualize_clusters_3d(tsne_embeddings, labels, "3D t-SNE Visualization of ClinicalBERT Semantic Clusters", "tsne_3d_clusters.png")
    visualize_pca(embeddings, labels)

    generate_wordclouds(data, labels)
    save_labeled_dataset(data, labels)

if __name__ == "__main__":
    main()