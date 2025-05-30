from collections import Counter
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Read text data
output = "unsupervised_model/kmeans-rulebased/output"
os.makedirs(output, exist_ok=True)
filename = "annotated_data/for-unsupervised/sentences.csv"
df = pd.read_csv(filename)
raw_sentences = df.iloc[:, 0].dropna().astype(str).tolist()

# Preprocessing: lowercase, remove symbols, collapse whitespace
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

preprocessed_sentences = [preprocess_text(s) for s in raw_sentences]

# Define ICF category keywords and phrases
lemmatizer = WordNetLemmatizer()
def simple_tokenize(text): return re.findall(r'\b\w+\b', text.lower())

category_dict = {
    1: {"phrases": ["lying down", "squatting", "kneeling", "sitting", "standing", "bending", "rolling over",
                    "shift body center of gravity", "maintain lying position", "maintain sitting position",
                    "maintain standing position", "transferring oneself", "lifting", "walking", "running",
                    "climbing", "crawling", "driving", "riding animals", "carrying", "pushing", "pulling",
                    "ambulate", "use a cane", "walk independently", "independent mobility", "walking aid",
                    "get out of bed", "stand", "gait"],
        "keywords": ["lying", "squat", "kneel", "sit", "stand", "bend", "roll", "shift", "maintain",
                     "transfer", "lift", "walk", "run", "climb", "crawl", "drive", "ride", "carry",
                     "push", "pull", "ambulate", "cane", "independent", "aid", "bed", "gait"]},
    2: {"phrases": ["washing", "bathing", "toileting", "dressing", "eating", "drinking", "grooming",
                    "preparing meals", "shopping", "cleaning", "housework", "caring for household",
                    "disposing garbage", "maintaining appliances", "managing health", "assisting others"],
        "keywords": ["wash", "bathe", "toilet", "dress", "eat", "drink", "groom", "prepare", "shop",
                     "clean", "housework", "care", "dispose", "maintain", "health", "assist"]},
    3: {"phrases": ["respect in relationships", "appreciation", "tolerance", "criticism", "physical contact",
                    "forming relationships", "terminating relationships", "interacting", "social cues",
                    "family relationships", "romantic relationships", "spousal relationships", "sexual relationships",
                    "siblings", "friends", "neighbors", "conversation"],
        "keywords": ["respect", "appreciate", "tolerate", "criticize", "contact", "form", "terminate",
                     "interact", "social", "family", "romantic", "spouse", "sexual", "sibling", "friend",
                     "neighbor", "conversation"]},
    4: {"phrases": ["watching", "listening", "touching", "smelling", "tasting", "learning",
                    "acquiring language", "reading", "writing", "attention", "solving problems",
                    "decision making", "communicating", "copying", "rehearsing", "focusing",
                    "thinking", "speaking"],
        "keywords": ["watch", "listen", "touch", "smell", "taste", "learn", "acquire", "language",
                     "read", "write", "attention", "solve", "decide", "communicate", "copy",
                     "rehearse", "focus", "think", "speak"]}
}

lemmatized_keywords = {
    cat: {
        "phrases": set(d["phrases"]),
        "keywords": set(lemmatizer.lemmatize(w) for w in d["keywords"])
    }
    for cat, d in category_dict.items()
}

#  Rule-based classification loop
labels, token_lengths = [], []
for orig, sent in zip(raw_sentences, preprocessed_sentences):
    tokens = simple_tokenize(sent)
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    token_lengths.append(len(lemmas))
    s_text = " ".join(tokens)
    s_lemmas = set(lemmas)
    best_cat, phrase_hits, word_hits = 0, 0, 0
    for cat, entry in lemmatized_keywords.items():
        ph = sum(1 for phrase in entry["phrases"] if phrase in s_text)
        wh = sum(1 for word in entry["keywords"] if word in s_lemmas)
        if ph >= 1 or wh >= 1:
            best_cat = cat
            phrase_hits = ph
            word_hits = wh
            break
    labels.append(best_cat if (phrase_hits >= 1 or word_hits >= 1) else 0)

#  BioBERT Embeddings + KMeans
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
embeddings = model.encode(preprocessed_sentences, batch_size=32, show_progress_bar=True)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

#  Auto-label based on cluster purity
final_labels = labels.copy()
for c in range(10):
    idx = [i for i, cl in enumerate(cluster_labels) if cl == c]
    cnt = Counter(labels[i] for i in idx if labels[i] != 0)
    if cnt and len(idx) >= 30:
        top_cat, count = cnt.most_common(1)[0]
        purity = count / len(idx)
        if purity >= 0.80:
            for i in idx:
                if final_labels[i] == 0 and token_lengths[i] >= 5:
                    final_labels[i] = top_cat

#  Print per-cluster summary
label_names = {
    0: "Unlabelled",
    1: "Mobility",
    2: "SC&DL",
    3: "IPIR",
    4: "COM&COG"
}

for c in range(10):
    idx = [i for i, cl in enumerate(cluster_labels) if cl == c]
    labeled_idx = [i for i in idx if labels[i] != 0]
    cnt = Counter(labels[i] for i in labeled_idx)

    print(f"\nCluster {c}")
    print(f"  Total sentences: {len(idx)}")
    print(f"  Labeled sentences: {len(labeled_idx)}")
    if cnt:
        top_cat, top_count = cnt.most_common(1)[0]
        purity = top_count / len(labeled_idx)
        print(f"  Dominant label: {label_names[top_cat]}")
        print(f"  Purity among labeled: {purity:.2%}")
        print(f"  Label distribution: {{ {', '.join(f'{label_names[k]}: {v}' for k, v in cnt.items())} }}")
    else:
        print("  No labeled sentences in this cluster.")

#  UMAP 3D Visualization
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
reduced_3d = reducer.fit_transform(embeddings)

#  3D plot by cluster ID
cluster_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
                  '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#17becf']
views = [(30, 30), (30, 150), (60, 240)]
fig = plt.figure(figsize=(18, 6))
fig.suptitle("3D UMAP – Colored by KMeans Clusters", fontsize=16)
for i, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    for c in range(10):
        ix = [j for j, cl in enumerate(cluster_labels) if cl == c]
        ax.scatter(reduced_3d[ix, 0], reduced_3d[ix, 1], reduced_3d[ix, 2],
                   label=f"Cluster {c}", alpha=0.7, s=10, color=cluster_colors[c % len(cluster_colors)])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"View {i+1} (Elev={elev}°, Azim={azim}°)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    if i == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10, title="Cluster ID")
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output, "3d_umap_clusters.png"), dpi=300)

#  3D plot by final labels
label_palette = {
    0: "#d3d3d3", 1: "#00cc44", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"
}
fig = plt.figure(figsize=(18, 6))
fig.suptitle("3D UMAP – Colored by Final ICF Labels", fontsize=16)
for i, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    for lbl in sorted(set(final_labels)):
        ix = [j for j, lab in enumerate(final_labels) if lab == lbl]
        ax.scatter(reduced_3d[ix, 0], reduced_3d[ix, 1], reduced_3d[ix, 2],
                   label=f"{label_names[lbl]}", alpha=0.7, s=10,
                   color=label_palette.get(lbl, "#000000"))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"View {i+1} (Elev={elev}°, Azim={azim}°)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    if i == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10, title="ICF Category")
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output, "3d_umap_clusters_labels.png"), dpi=300)

#  Label count bar chart
label_counts = Counter(final_labels)
labels_sorted = sorted(label_counts.keys())
heights = [label_counts[k] for k in labels_sorted]
colors = [label_palette.get(k, "black") for k in labels_sorted]


plt.figure(figsize=(10, 6))  # wider figure
bars = plt.bar([label_names[k] for k in labels_sorted], heights, color=colors, log=True)

# Add value labels on top of each bar
for bar, height in zip(bars, heights):
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=30, ha='right')  # rotate labels
plt.title("Final Label Distribution (Log Scale)")
plt.xlabel("Label")
plt.ylabel("Count (log scale)")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output, "3d_umap_clusters.png"), dpi=300)


#  Export results to Excel
output_df = pd.DataFrame({
    "Original Sentence": raw_sentences,
    "Preprocessed": preprocessed_sentences,
    "Rule-Based Label": [label_names[l] for l in labels],
    "Final Label": [label_names[l] for l in final_labels],
    "Cluster ID": cluster_labels,
    "Token Count": token_lengths
})
output_df.to_csv(os.path.join(output, "fsi_labeled_results.csv"), index=False)