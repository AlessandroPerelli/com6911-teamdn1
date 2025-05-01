import os
import random
import itertools
import logging

import numpy as np
import pandas as pd
import scipy
import scipy.linalg

if not hasattr(scipy.linalg, 'triu'):
    import numpy as _np
    scipy.linalg.triu = lambda m, k=0: _np.triu(m, k=k)

from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH     = 'C:/Users/molly/COM6911/com6911-teamdn1/annotated_data/full/combined_complete.xlsx'
FASTTEXT_PATH = 'C:/Users/molly/COM6911/cc.en.300.vec'
CACHE_PATH    = 'embedding_matrix.npy'
MAX_VOCAB     = 20000
MAX_LEN       = 100
LABEL_MAP     = {
    1: 'mobility',
    2: 'self_care_domestic_life',
    3: 'interpersonal_interactions',
    4: 'communication_cognition'
}

logger.info("Loading data from %s", DATA_PATH)
df = pd.read_excel(DATA_PATH, engine='openpyxl')

def parse_labels(entry):
    parts = str(entry).split(',')
    return [
        int(p.strip().strip('"').strip("'"))
        for p in parts
        if p.strip().strip('"').strip("'").isdigit() and
           int(p.strip().strip('"').strip("'")) in LABEL_MAP
    ]

df['labels_list'] = df['label'].apply(parse_labels)
mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
Y = mlb.fit_transform(df['labels_list'])
X_sentences = df['sentence'].astype(str).tolist()

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_sentences, Y, test_size=0.2,
    random_state=RANDOM_SEED, stratify=Y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.2,
    random_state=RANDOM_SEED, stratify=y_tmp
)
logger.info("Data splits — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(text):
    return [tok.lower() for tok in word_tokenize(text)]

X_train_tok = [tokenize(s) for s in X_train]
X_val_tok   = [tokenize(s) for s in X_val]
X_test_tok  = [tokenize(s) for s in X_test]

logger.info("Building vocabulary from training data")
counter = {}
for sent in X_train_tok:
    for tok in sent:
        counter[tok] = counter.get(tok, 0) + 1
most_common = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB-2]
word2idx = {w: i+2 for i, (w, _) in enumerate(most_common)}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1
vocab_size = len(word2idx)

if os.path.exists(CACHE_PATH):
    logger.info("Loading embedding matrix from cache %s", CACHE_PATH)
    embedding_matrix = np.load(CACHE_PATH)
    EMB_DIM = embedding_matrix.shape[1]
else:
    logger.info("Loading FastText vectors from %s", FASTTEXT_PATH)
    ft = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)
    EMB_DIM = ft.vector_size
    embedding_matrix = np.zeros((vocab_size, EMB_DIM), dtype=np.float32)
    for w, idx in word2idx.items():
        embedding_matrix[idx] = ft[w] if w in ft else np.random.normal(scale=0.6, size=(EMB_DIM,))
    np.save(CACHE_PATH, embedding_matrix)
    logger.info("Saved embedding matrix to %s", CACHE_PATH)

logger.info("Encoding and padding sequences to length %d", MAX_LEN)
def encode(tokens):
    idxs = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    if len(idxs) < MAX_LEN:
        idxs += [word2idx['<PAD>']] * (MAX_LEN - len(idxs))
    else:
        idxs = idxs[:MAX_LEN]
    return idxs

X_tr_seq  = np.array([encode(s) for s in X_train_tok])
X_val_seq = np.array([encode(s) for s in X_val_tok])
X_te_seq  = np.array([encode(s) for s in X_test_tok])

def get_loader(X_seq, Y_arr, batch_size, shuffle=False):
    ds = TensorDataset(
        torch.LongTensor(X_seq),
        torch.FloatTensor(Y_arr)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes,
                 filter_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False,
            padding_idx=word2idx['<PAD>']
        )
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, emb_dim))
            for fs in filter_sizes
        ])
        self.bns1 = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv2d(num_filters, num_filters, (3,1), padding=(1,0))
            for _ in filter_sizes
        ])
        self.bns2 = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)    
        pooled = []
        for conv1, bn1, conv2, bn2 in zip(self.convs1, self.bns1,
                                          self.convs2, self.bns2):
            y = torch.relu(bn1(conv1(x)))
            y = torch.relu(bn2(conv2(y)))
            y = y.squeeze(3)  
            y = torch.max(y, dim=2)[0]
            pooled.append(y)
        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        return torch.sigmoid(self.fc(out))

param_grid = {
    'lr': [1e-3, 1e-4],
    'batch_size': [32, 64],
    'num_filters': [100, 150],
    'dropout': [0.3, 0.5]
}

best_score, best_cfg = 0.0, None
for lr, bs, nf, dp in itertools.product(*param_grid.values()):
    logger.info("Grid search config: lr=%g, bs=%d, filters=%d, dp=%.2f", lr, bs, nf, dp)
    model = TextCNN(vocab_size, EMB_DIM, len(LABEL_MAP), num_filters=nf, dropout=dp).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    tr_loader = get_loader(X_tr_seq, y_train, batch_size=bs, shuffle=True)
    vl_loader = get_loader(X_val_seq, y_val, batch_size=bs, shuffle=False)
    for _ in range(5):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimiser.step()
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in vl_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            p = (model(Xb) >= 0.5).float().cpu().numpy()
            all_p.append(p)
            all_t.append(yb.cpu().numpy())
    all_p, all_t = np.vstack(all_p), np.vstack(all_t)
    score = f1_score(all_t, all_p, average='micro')
    logger.info("Val micro-F1 = %.4f", score)
    if score > best_score:
        best_score, best_cfg = score, (lr, bs, nf, dp)

logger.info("Best config: %s -> micro-F1 %.4f", best_cfg, best_score)

lr, bs, nf, dp = best_cfg
model = TextCNN(vocab_size, EMB_DIM, len(LABEL_MAP), num_filters=nf, dropout=dp).to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

X_full_seq = np.vstack([X_tr_seq, X_val_seq])
y_full = np.vstack([y_train, y_val])
full_loader = get_loader(X_full_seq, y_full, batch_size=bs, shuffle=True)
test_loader = get_loader(X_te_seq, y_test, batch_size=bs, shuffle=False)

for epoch in range(10):
    model.train()
    for Xb, yb in full_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimiser.step()

model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        p = (model(Xb) >= 0.5).float().cpu().numpy()
        all_p.append(p)
        all_t.append(yb.cpu().numpy())
test_preds = np.vstack(all_p)
test_targs = np.vstack(all_t)
test_micro = f1_score(test_targs, test_preds, average='micro')
test_macro = f1_score(test_targs, test_preds, average='macro')

logger.info("Final Test micro-F1: %.4f", test_micro)
logger.info("Final Test macro-F1: %.4f", test_macro)

cm = multilabel_confusion_matrix(test_targs, test_preds)
for i, cls in enumerate(LABEL_MAP.values()):
    tn, fp, fn, tp = cm[i].ravel()
    logger.info("%s — TP:%d FP:%d FN:%d TN:%d", cls, tp, fp, fn, tn)

logger.info("Classification Report:\n%s", classification_report(test_targs, test_preds, target_names=list(LABEL_MAP.values())))
