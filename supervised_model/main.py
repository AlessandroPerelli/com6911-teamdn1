import os
import random
import logging
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report

# configuration
data_path = r'C:/Users/molly/COM6911/com6911-teamdn1/annotated_data/full/combined_complete.xlsx'
fasttext_path = r'C:/Users/molly/COM6911/cc.en.300.vec'
random_seed = 44
embedding_cache = 'embedding_matrix.npy'

# hyperparameters
MAX_VOCAB = 1000
MAX_LEN = 100
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
RNN_HIDDEN = 128
DROPOUT = 0.5
LR = 0.001
BATCH_SIZE = 32
NUM_CLASSES = 4
MAX_EPOCHS = 20
PATIENCE = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

label_map = {
    1: 'mobility',
    2: 'self_care_domestic_life',
    3: 'interpersonal_interactions',
    4: 'communication_cognition'
}

# load and preprocess data
df = pd.read_excel(data_path)

def parse_labels(item):
    parts = str(item).split(',')
    return [int(p.strip().strip('"').strip("'"))
            for p in parts if p.strip().strip('"').strip("'").isdigit() and int(p.strip().strip('"').strip("'")) in label_map]

# filter positives and under-sample negatives
df['labels'] = df['label'].apply(parse_labels)

# sanity check: raw label vs parsed labels
print("RAW VS PARSED LABELS")
pos_df = df[df['labels'].map(len) > 0]
for i, row in pos_df.head(10).iterrows():
    raw_label = row['label']
    parsed    = parse_labels(raw_label)
    print(f"{i}) Raw label = {raw_label!r} Parsed label= {parsed}")
    print(f"  Sentence: {row['sentence']!r}\n")

pos_df = df[df['labels'].map(len) > 0]
neg_df = df[df['labels'].map(len) == 0]
neg_sample = neg_df.sample(frac=0.05, random_state=random_seed)
df = pd.concat([pos_df, neg_sample]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

# binarise labels and split
mlb = MultiLabelBinarizer(classes=list(label_map.keys()))
y = mlb.fit_transform(df['labels'])
X = df['sentence'].astype(str).tolist()
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=random_seed, stratify=y_tmp)

# tokenisation and build vocabulary
def tokenize(text): return [tok.lower() for tok in word_tokenize(text, preserve_line=True)]

X_train_tok = [tokenize(s) for s in X_train]
X_val_tok   = [tokenize(s) for s in X_val]
X_test_tok  = [tokenize(s) for s in X_test]

counter = {}
for tok in X_train_tok:
    for t in tok:
        counter[t] = counter.get(t, 0) + 1
most_common = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB-2]
word2idx = {w: i+2 for i, (w, _) in enumerate(most_common)}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1
vocab_size = len(word2idx)

# load/build embedding matrix
if os.path.exists(embedding_cache):
    embedding_matrix = np.load(embedding_cache)
else:
    ft = KeyedVectors.load_word2vec_format((fasttext_path), binary=False)
    emb_dim = ft.vector_size
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for w, idx in word2idx.items():
        embedding_matrix[idx] = ft[w] if w in ft else np.random.normal(scale=0.6, size=(emb_dim,))
    np.save(embedding_cache, embedding_matrix)

# sequence encoding
def encode(tokens):
    idxs = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    return idxs[:MAX_LEN] + [word2idx['<PAD>']] * max(0, MAX_LEN - len(idxs))

X_tr_seq  = np.array([encode(s) for s in X_train_tok])
X_val_seq = np.array([encode(s) for s in X_val_tok])
X_te_seq  = np.array([encode(s) for s in X_test_tok])

# model definition
class CNNRNNHybrid(nn.Module):
    def __init__(self, vocab_size, emb_matrix, filter_sizes, num_filters, rnn_hidden, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix), freeze=False, padding_idx=word2idx['<PAD>'])
        self.convs = nn.ModuleList([nn.Conv1d(emb_matrix.shape[1], num_filters, fs)
                                    for fs in filter_sizes])
        self.gru   = nn.GRU(num_filters * len(filter_sizes), rnn_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        conv_outs = [F.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        cat = torch.cat(conv_outs, dim=1).unsqueeze(1)
        _, h = self.gru(cat)
        return torch.sigmoid(self.fc(self.dropout(h.squeeze(0))))

# data loaders
d_train = TensorDataset(torch.LongTensor(X_tr_seq), torch.FloatTensor(y_train))
d_val   = TensorDataset(torch.LongTensor(X_val_seq), torch.FloatTensor(y_val))
d_test  = TensorDataset(torch.LongTensor(X_te_seq), torch.FloatTensor(y_test))
loader_train = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
loader_val   = DataLoader(d_val,   batch_size=BATCH_SIZE)
loader_test  = DataLoader(d_test,  batch_size=BATCH_SIZE)

# initialise model, loss and optimiser
model = CNNRNNHybrid(vocab_size, embedding_matrix, FILTER_SIZES, NUM_FILTERS, RNN_HIDDEN, DROPOUT, NUM_CLASSES).to(device)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=LR)

# training loop
best_f1 = 0.0
no_improve = 0
best_state = None
for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    for xb, yb in loader_train:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimiser.step()

    # validation
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in loader_val:
            xb, yb = xb.to(device), yb.to(device)
            p = (model(xb) >= 0.5).float().cpu().numpy()
            all_p.append(p); all_t.append(yb.cpu().numpy())
    all_p = np.vstack(all_p); all_t = np.vstack(all_t)
    val_f1 = f1_score(all_t, all_p, average='micro')
    logging.info(f"Epoch {epoch}: Val micro-F1 = {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve = 0
        best_state = model.state_dict()
    else:
        no_improve += 1
    if no_improve >= PATIENCE:
        logging.info("Early stopping triggered")
        break

# load best model and evaluate on test set
model.load_state_dict(best_state)
model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for xb, yb in loader_test:
        xb, yb = xb.to(device), yb.to(device)
        p = (model(xb) >= 0.5).float().cpu().numpy()
        all_p.append(p); all_t.append(yb.cpu().numpy())
all_p = np.vstack(all_p); all_t = np.vstack(all_t)

# metrics
micro = f1_score(all_t, all_p, average='micro')
macro = f1_score(all_t, all_p, average='macro')
logging.info(f"Test micro-F1: {micro:.4f}")
logging.info(f"Test macro-F1: {macro:.4f}")

cm = multilabel_confusion_matrix(all_t, all_p)
for i, cls in enumerate(label_map.values()):
    tn, fp, fn, tp = cm[i].ravel()
    logging.info(f"{cls}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# error analysis: five misclassified examples per class
for class_idx, class_name in enumerate(label_map.values(), start=1):
    mis_idx = [i for i in range(len(all_t)) if all_t[i, class_idx-1] != all_p[i, class_idx-1]]
    print(f"Misclassified examples for class {class_idx} ({class_name})")
    for idx in mis_idx[:5]:
        sentence = X_test[idx]
        true_flag = all_t[idx, class_idx-1]
        pred_flag = all_p[idx, class_idx-1]
        true_label = class_idx if true_flag == 1 else 0
        pred_label = class_idx if pred_flag == 1 else 0
        print(f"""Sentence: {sentence}
  True label: {true_label}
  Pred label: {pred_label}
""")

# multi-label predictions: examples classified with more than one class
multi_idx = [i for i in range(len(all_p)) if all_p[i].sum() > 1]
print("Examples predicted as multiple classes:")
for idx in multi_idx[:5]:
    sentence = X_test[idx]
    pred_flags = all_p[idx]
    true_flags = all_t[idx]
    pred_indices = [str(i+1) for i, flag in enumerate(pred_flags) if flag == 1]
    true_indices = [str(i+1) for i, flag in enumerate(true_flags) if flag == 1]
    print(f"""Sentence: {sentence}
  True class: {', '.join(true_indices) if true_indices else '0'}        
  Predicted class: {', '.join(pred_indices)}
""")
