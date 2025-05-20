"""
Functional Status Information analysis with unsupervised clustering methods for unlabeled data.
Author: Tara Jain
Student ID: acp24tkj
Date: 19/05/2025

Description:
This script performs an unsupervised clustering analysis with LDA and Kmeans on the MIMIC IV dataset with Pyspark intended to ensure scalability.

Requirements:
- PySpark
- Matplotlib
- Spacy
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, collect_list, concat_ws, udf
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer
from wordcloud import WordCloud

DATA_PATH = "data/labeled_starter.csv"
SEED = 314159
OUTPUTS_PATH = "Outputs/"
# Configure logger
os.makedirs(OUTPUTS_PATH, exist_ok=True)
log_file_path = os.path.join(OUTPUTS_PATH, "logs.txt")
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
os.makedirs("Outputs", exist_ok=True)

def init_spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("Fsi-nlp") \
        .getOrCreate()

# Load spaCy once globally
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Basic English stopwords from spaCy
stopwords = nlp.Defaults.stop_words

@udf(ArrayType(StringType()))
def tokenize_filter_lemmatize(sentence: str):
    """
    Processes a given sentence by tokenizing, filtering, and lemmatizing its words.
    This function performs the following steps:
    1. Tokenizes the input sentence using a natural language processing (NLP) model.
    2. Filters out tokens that are not alphabetic, are in the stopwords list, or have a lemma length of 2 or less.
    3. Converts the remaining tokens to their lowercase lemmatized forms.
    """

    if not sentence:
        return []
    doc = nlp(sentence)
    tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and token.text.lower() not in stopwords and len(token.lemma_) > 2
    ]
    return tokens if tokens else None

def preprocess_sentences(unprocessed_sentences: DataFrame) -> DataFrame:
    """
    Preprocess sentences using spaCy:
    - Lowercase
    - Remove punctuation/non-alpha characters
    - Remove stopwords
    - Apply lemmatization
    - Drop rows with empty token lists
    Returns a DataFrame with 'filtered_sentences' and 'tokens'.
    """
    logging.info("Preprocessing sentences with spaCy NLP...")

    processed_sentences = unprocessed_sentences.withColumn(
        "tokens",
        tokenize_filter_lemmatize(col("sentence"))
    ).filter(
        col("tokens").isNotNull()
    ).cache()

    logging.info(f"Number of processed sentences: {processed_sentences.count()}")

    return processed_sentences

def get_clinicalbert_embeddings(tokens_list):
    """
    Load the ClinicalBERT model and tokenizer from Hugging Face.
    This function returns a function that takes a list of tokens and returns their embeddings.
    """
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def embed(tokens):
        text = " ".join(tokens)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return embed

def extract_features(sentences, method="tf", vocab_size: int = 5000, min_df: int = 2):
    """
    Extract features from sentences using either TF-IDF or ClinicalBERT embeddings.
    """
    if method == "tf":
        vectorizer = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=vocab_size, minDF=min_df)
        model = vectorizer.fit(sentences)
        vectorized_data = model.transform(sentences)
        vectorized_data.cache()
        return vectorized_data, model.vocabulary

    elif method == "clinical_bert":
        embed = get_clinicalbert_embeddings([])
        get_bert_embedding_udf = udf(lambda tokens: embed(tokens), ArrayType(DoubleType()))
        embedded_data = sentences.withColumn("features", get_bert_embedding_udf(col("tokens")))
        return embedded_data, None

def kmeans(data: DataFrame, k: int):
    """
    Perform KMeans clustering on a DataFrame that contains a 'features' column.
    """
    logging.info(f"Running KMeans with k={k}")

    to_vector_udf = udf(lambda arr: Vectors.dense([float(x) for x in arr]), VectorUDT())
    vectorized_data = data.withColumn("features_vec", to_vector_udf(col("features")))

    kmeans = KMeans(featuresCol="features_vec", predictionCol=f"prediction_{k}", k=k, seed=SEED)
    model = kmeans.fit(vectorized_data)
    clustered_data = model.transform(vectorized_data)
    return model, clustered_data

def lda(sentences: DataFrame, vocabulary, k: int = 5, max_iter: int = 10):
    """
    Perform LDA on a DataFrame that contains a 'features' column.
    """
    logging.info(f"Running LDA with k={k}, maxIter={max_iter}")

    lda_model = LDA(k=k, maxIter=max_iter, seed=SEED, featuresCol="features").fit(sentences)

    perplexity = lda_model.logPerplexity(sentences)

    topics = lda_model.describeTopics()
    vocab = vocabulary
    topic_words = topics.rdd.map(lambda row: [vocab[i] for i in row['termIndices']]).collect()

    for idx, words in enumerate(topic_words):
        logging.info(f"Topic {idx}: {', '.join(words)}")

    return lda_model, perplexity

def evaluate_clusters(clustered_data):
    """
    Evaluate the clustering using silhouette score.
    """
    pandas_df = clustered_data.select("features", "cluster").toPandas()
    X = pandas_df["features"].apply(lambda x: list(map(float, x))).tolist()

    y = pandas_df["cluster"].tolist()

    score = silhouette_score(X, y)
    logging.info(f"Silhouette Score: {score:.4f}")
    return score

def generate_wordclouds(clustered_data, cluster_col, output_dir):
    """
    Generate word clouds for each cluster in the clustered data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure necessary columns exist
    if not all(col_name in clustered_data.columns for col_name in ["tokens", cluster_col]):
        logging.error("Missing required columns in clustered_data. 'tokens' and cluster_col must be present.")
        return

    clustered_data = clustered_data.withColumn("text", concat_ws(" ", col("tokens")))

    grouped_data = clustered_data.groupBy(cluster_col).agg(
        concat_ws(" ", collect_list("text")).alias("cluster_text")
    )

    grouped_data_pd = grouped_data.toPandas()

    for _, row in grouped_data_pd.iterrows():
        cluster_label = row[cluster_col]
        cluster_text = row["cluster_text"]

        if not isinstance(cluster_text, str) or not cluster_text.strip():
            logging.warning(f"Empty or invalid text for cluster {cluster_label}. Skipping word cloud.")
            continue

        try:
            wordcloud = WordCloud(
                width=1600,
                height=800,
                background_color="white",
                collocations=False,
                colormap="tab10"
            ).generate(cluster_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.title(f"Wordcloud for Cluster {cluster_label}", fontsize=20)
            filepath = os.path.join(output_dir, f"wordcloud_cluster_{cluster_label}.png")
            plt.savefig(filepath, dpi=300)
            plt.close()
            logging.info(f"Saved wordcloud for cluster {cluster_label} to {filepath}")

        except Exception as e:
            logging.error(f"Failed to generate wordcloud for cluster {cluster_label}: {e}")

def dimensionality_reduction(clustered_data):
    """
    Perform dimensionality reduction using t-SNE.
    """
    pandas_df = clustered_data.select("features", "cluster").toPandas()
    X = np.array(pandas_df["features"].tolist())
    y = pandas_df["cluster"].tolist()
    tsne = TSNE(n_components=3, random_state=SEED)
    X_tsne = tsne.fit_transform(X)
    result_df = pd.DataFrame(X_tsne, columns=["x", "y", "z"])
    result_df["cluster"] = y
    return result_df

def visualize_clusters_and_save(reduced_df, k, filename="Outputs/tsne_clusters.csv", algorithm="KMeans"):
    """
    Visualize clusters in 3D using t-SNE and save the figure.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_df['x'], reduced_df['y'], reduced_df['z'], c=reduced_df['cluster'], cmap='tab10')
    ax.set_title("t-SNE Clustering Visualization")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig(f"Outputs/tsne_clusters_k={k}_{algorithm}.png")
    plt.close()
    reduced_df.to_csv(filename, index=False)
    logging.info(f"Saved 3D cluster visualization data to {filename}")


def main():

    spark = init_spark()
    # sentences = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    
    # Read in only the first 2000 records for development and test.
    sentences = spark.read.csv(DATA_PATH, header=True, inferSchema=True).limit(1000)

    logging.info(f"Number of sentences in dataframe: {sentences.count()}")

    sentences = preprocess_sentences(sentences)

    row_count = sentences.count()
    logging.info(f"[DEBUG] Row count before writing CSV: {row_count}")
        # Extract term frequency features
    sentences_with_features, lda_vocab = extract_features(sentences, vocab_size=5000, min_df=2)

    # Try multiple values of k for LDA and compare perplexity
    get_dominant_topic = udf(lambda dist: int(np.argmax(dist)), IntegerType())

    results = []

    for k in [4, 5, 6]:
        model, perplexity = lda(sentences_with_features, lda_vocab, k=k)
        transformed = model.transform(sentences_with_features)
        transformed = transformed.withColumn("features", col("topicDistribution"))
        transformed = transformed.withColumn("cluster", get_dominant_topic(col("topicDistribution")))

        reduced_df = dimensionality_reduction(transformed)

        visualize_clusters_and_save(reduced_df, k=k, filename=f"Outputs/tsne_clusters_k={k}_LDA.csv", algorithm="LDA")
        
        generate_wordclouds(transformed.select("tokens", "cluster"), "cluster", output_dir=f"Outputs/wordclouds_lda_k={k}")

        silhouette = evaluate_clusters(transformed.select("features", "cluster"))

        results.append({"algorithm": "LDA", "k": k, "perplexity": perplexity, "silhouette": silhouette})

    # pd.DataFrame(results).to_csv("Outputs/clustering_evaluation.csv", index=False)

    # Convert to pandas dataframd and save to CSV for local testing
    # and debugging
    # sentences_pd = sentences.toPandas()
    # sentences_pd.to_csv("Outputs/sentences_preprocessed.csv", index=False)
    # logging.info("Saved processed sentences to Outputs/sentences_preprocessed.csv")


    # TODO(P2 feature): Perform NER to remove medication names or use a list. 

    ## K-means ##

    sentences_with_embeddings, _ = extract_features(sentences, method = "clinical_bert")

    kmeans_models = []
    k_cluster_labels = []

    logging.info("Beginning KMeans clustering.")
    # Try multiple values of k for Kmeans.
    for k in [3, 4, 5]:

        k_model, clustered_data = kmeans(sentences_with_embeddings, k=k)
        kmeans_models.append(k_model)

        k_cluster_labels.append(clustered_data)

        logging.info(f"Columns in clustered_data: {clustered_data.columns}")

        # Save the clustered data with cluster labels
        clustered_data = clustered_data.select("sentence", "tokens", "features", col(f"prediction_{k}").alias("cluster"))

        # Save to CSV for local testing and inspection.
        # clustered_data_pd = clustered_data.toPandas()  # Convert to pandas DataFrame
        # clustered_data_pd.to_csv(f"Outputs/kmeans_clusters_k{k}.csv", index=False)  # Save to CSV
        # logging.info(f"Saved KMeans clusters with k={k} to Outputs/kmeans_clusters_k{k}.csv")

        reduced_df = dimensionality_reduction(clustered_data)

        visualize_clusters_and_save(reduced_df, k,filename=f"Outputs/tsne_clusters_k={k}.csv")
        
        # evaluate_clusters(clustered_data)
        
        generate_wordclouds(clustered_data.select("sentence", "cluster"), "cluster", output_dir="Outputs/wordclouds/")
    
        silhouette = evaluate_clusters(clustered_data.select("features", "cluster"))

        results.append({"algorithm": "KMeans", "k": k, "silhouette": silhouette})

    pd.DataFrame(results).to_csv("Outputs/clustering_evaluation.csv", index=False)

    # evaluate_clusters()
    # visualize_clusters_and_save()
    # dimensionality_reduction()

    # unpersist
    sentences.unpersist()
    sentences_with_features.unpersist()
    sentences_with_embeddings.unpersist()
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
    