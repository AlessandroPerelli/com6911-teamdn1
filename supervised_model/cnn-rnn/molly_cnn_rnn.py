import os
import random
import logging
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
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
from sklearn.metrics import f1_score, multilabel_confusion_matrix

# configuration
data_path = 'annotated_data/full/combined_complete.csv'
output_dir = 'supervised_model/cnn-rnn/output'
os.makedirs(output_dir, exist_ok=True)
fasttext_path = 'fasttext/cc.en.300.vec'
random_seed = 44
embedding_cache = 'embedding_matrix.npy'

# hyperparameters
MAX_VOCAB = 1000
MAX_LEN = 100
FILTER_SIZE = 2 # optimised between (1-7)
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
df = pd.read_csv(data_path, dtype=str)

def parse_labels(item):
    parts = str(item).split(',')
    return [int(p.strip().strip('"').strip("'"))
            for p in parts if p.strip().strip('"').strip("'").isdigit() and int(p.strip().strip('"').strip("'")) in label_map]

# under-sample negatives
df['labels'] = df['label'].apply(parse_labels)

# check samples before undersampling
total_before   = df.shape[0]
pos_before     = df[df['labels'].map(len) > 0].shape[0]
neg_before     = df[df['labels'].map(len) == 0].shape[0]
print(f"Before undersampling: total = {total_before}, positives = {pos_before}, negatives = {neg_before}")

pos_df = df[df['labels'].map(len) > 0]
neg_df = df[df['labels'].map(len) == 0]
neg_sample = neg_df.sample(frac=0.05, random_state=random_seed)
df = pd.concat([pos_df, neg_sample]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

# check samples after undersampling
total_after   = df.shape[0]
pos_after     = df[df['labels'].map(len) > 0].shape[0]
neg_after     = df[df['labels'].map(len) == 0].shape[0]
print(f"After undersampling: total = {total_after}, positives = {pos_after}, negatives = {neg_after}")

# binarise labels and split
mlb = MultiLabelBinarizer(classes=list(label_map.keys()))
y = mlb.fit_transform(df['labels'])
X = df['sentence'].astype(str).tolist()
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=random_seed, stratify=y_tmp)

# check data split
splits = {
    'train': y_train,
    'validation': y_val,
    'test': y_test
}
records = []
for name, y in splits.items():
    total = y.shape[0]
    class_counts = y.sum(axis=0)
    negatives = np.sum(y.sum(axis=1) == 0)
    records.append({
        'split': name,
        'total': total,
        'class_1': int(class_counts[0]),
        'class_2': int(class_counts[1]),
        'class_3': int(class_counts[2]),
        'class_4': int(class_counts[3]),
        'no_label (0)': negatives
    })
df_counts = pd.DataFrame(records).set_index('split')
print(df_counts)

# tokenisation and build vocab
def tokenise(text):
    return [tok.lower() for tok in word_tokenize(text, preserve_line=True)]

X_train_tok = [tokenise(s) for s in X_train]
X_val_tok = [tokenise(s) for s in X_val]
X_test_tok = [tokenise(s) for s in X_test]

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

X_tr_seq = np.array([encode(s) for s in X_train_tok])
X_val_seq = np.array([encode(s) for s in X_val_tok])
X_te_seq = np.array([encode(s) for s in X_test_tok])

# model definition
class CNNRNNHybrid(nn.Module):
    def __init__(self, emb_matrix, filter_size, num_filters, rnn_hidden, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix), freeze=False, padding_idx=word2idx['<PAD>'])
        self.conv = nn.Conv1d(emb_matrix.shape[1], num_filters, filter_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.gru = nn.GRU(num_filters, rnn_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        c = F.relu(self.conv(emb))
        p = self.pool(c).squeeze(-1).unsqueeze(1)
        _, h = self.gru(p)
        out = self.fc(self.dropout(h.squeeze(0)))
        return torch.sigmoid(out)

# data loaders
d_train = TensorDataset(torch.LongTensor(X_tr_seq), torch.FloatTensor(y_train))
d_val = TensorDataset(torch.LongTensor(X_val_seq), torch.FloatTensor(y_val))
d_test = TensorDataset(torch.LongTensor(X_te_seq), torch.FloatTensor(y_test))
loader_train = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(d_val, batch_size=BATCH_SIZE)
loader_test = DataLoader(d_test, batch_size=BATCH_SIZE)

# initialise model, loss and optimiser
model = CNNRNNHybrid(embedding_matrix, FILTER_SIZE, NUM_FILTERS, RNN_HIDDEN, DROPOUT, NUM_CLASSES).to(device)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=LR)

# training loop
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []
best_f1 = 0.0
no_improve = 0
best_state = None
for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for xb, yb in loader_train:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * xb.size(0)
        all_preds.append((preds >= 0.5).cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    epoch_loss = total_loss / len(d_train)
    train_losses.append(epoch_loss)
    train_f1 = f1_score(np.vstack(all_targets), np.vstack(all_preds), average='micro')
    train_f1s.append(train_f1)

    # validation
    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader_val:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            all_preds.append((preds >= 0.5).cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    epoch_val_loss = val_loss / len(d_val)
    val_losses.append(epoch_val_loss)
    preds_val = np.vstack(all_preds)
    targets_val = np.vstack(all_targets)
    val_f1 = f1_score(targets_val, preds_val, average='micro')
    val_f1s.append(val_f1)

    logging.info(
        f"Epoch {epoch}: "
        f"Train Loss={epoch_loss:.4f}, "
        f"Val Loss={epoch_val_loss:.4f}, "
        f"Train F1={train_f1:.4f}, "
        f"Val F1={val_f1:.4f}"
    )
    
    # early stopping
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
        xb = xb.to(device)
        all_p.append(model(xb).cpu().numpy())
        all_t.append(yb.numpy())

y_pred_probs = np.vstack(all_p)
y_test_array = np.vstack(all_t)
y_pred_bin = (y_pred_probs >= 0.5).astype(int)

# compute final metrics
test_micro = f1_score(y_test_array, (y_pred_probs >= 0.5).astype(int), average='micro')
test_macro = f1_score(y_test_array, (y_pred_probs >= 0.5).astype(int), average='macro')
logging.info(f"Test micro-F1: {test_micro:.4f}")
logging.info(f"Test macro-F1: {test_macro:.4f}")

# print evaluation metrics
mcm = multilabel_confusion_matrix(y_test_array, y_pred_bin)
for i, cm in enumerate(mcm, start=1):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Class {i} ({label_map[i]}): TN={tn}, FP={fp}, FN={fn}, TP={tp}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")

# confusion matrix
num_classes = y_test_array.shape[1]
N = num_classes + 1
agg_conf0 = np.zeros((N, N), dtype=int)
for true_row, pred_row in zip(y_test_array, y_pred_bin):
    true_idxs = np.where(true_row == 1)[0] + 1
    pred_idxs = np.where(pred_row == 1)[0] + 1
    if true_idxs.size == 0:
        true_idxs = np.array([0])
    if pred_idxs.size == 0:
        pred_idxs = np.array([0])
    for t in true_idxs:
        for p in pred_idxs:
            agg_conf0[t, p] += 1
plt.figure(figsize=(6,6))
plt.imshow(agg_conf0, interpolation='nearest', cmap='Blues')
labels = ['No-label'] + [f'Class {i}' for i in range(1, num_classes+1)]
plt.xticks(range(N), labels, rotation=45)
plt.yticks(range(N), labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN-RNN Confusion Matrix')
plt.colorbar()
thresh = agg_conf0.max() / 2.0
for i in range(N):
    for j in range(N):
        count = agg_conf0[i, j]
        colour = 'white' if count > thresh else 'black'
        plt.text(j, i, str(count), ha='center', va='center', color=colour, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cnn-rnn_confusion_matrix.png'))
plt.close()

# learning curves
epochs = range(1, len(train_losses) + 1)
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.title('CNN-RNN Learning Curve: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cnn-rnn_learning_curve_loss.png'))

plt.figure()
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s, label='Val F1')
plt.title('CNN-RNN Learning Curve: F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cnn-rnn_learning_curve_f1.png'))

# error analysis: five misclassified examples per class
for class_idx, class_name in enumerate(label_map.values(), start=1):
    mismatches = np.where(y_test_array[:, class_idx-1] != y_pred_bin[:, class_idx-1])[0]
    print(f"Misclassified examples for class {class_idx} ({class_name}):")
    for idx in mismatches[:5]:
        sentence = X_test[idx]
        true_flag = y_test_array[idx, class_idx-1]
        pred_flag = y_pred_bin[idx, class_idx-1]
        true_cls = class_idx if true_flag == 1 else 0
        pred_cls = class_idx if pred_flag == 1 else 0
        print(
            f"Sentence: {sentence}\n"
            f"  True class: {true_cls}\n"
            f"  Pred class: {pred_cls}\n"
        )
