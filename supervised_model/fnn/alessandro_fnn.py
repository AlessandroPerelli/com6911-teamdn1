import os
import random
import logging

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix


# directories
DATA_PATH = 'annotated_data/full/combined_complete.csv'
FASTTEXT_PATH = 'cc.en.300.vec'
EMBEDDING_CACHE = 'embedding_matrix.npy'
RANDOM_SEED = 44

# hyperparameters - using same as CNN for comparison
MAX_VOCAB = 1000
MAX_LEN = 100
EMB_DROPOUT = 0.5
FNN_HIDDENS = [256, 128]
LR = 0.001
BATCH_SIZE = 32
NUM_CLASSES = 4
MAX_EPOCHS = 20
PATIENCE = 3

label_map = {
    1: 'mobility',
    2: 'self_care_domestic_life',
    3: 'interpersonal_interactions',
    4: 'communication_cognition'
}

# set seeds and device
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if GPU is available, use it
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logging.info(f'Using device: {device}')

# Load data and parse labels
def load_data():
    df = pd.read_csv(DATA_PATH)

    def parse_labels(item):
        parts = str(item).split(',')
        return [int(p.strip().strip('"').strip("'"))
                for p in parts
                if p.strip().strip('"').strip("'").isdigit()
                and int(p.strip().strip('"').strip("'")) in label_map]
    df['labels'] = df['label'].apply(parse_labels)
    pos_df = df[df['labels'].map(len) > 0] # positive samples
    neg_df = df[df['labels'].map(len) == 0] # negative samples
    neg_sample = neg_df.sample(frac=0.05, random_state=RANDOM_SEED) # sample negative samples
    df2 = pd.concat([pos_df, neg_sample]).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    X = df2['sentence'].astype(str).tolist()
    y = df2['labels'].tolist()
    return X, y # sentences and labels


# tokenise sentences and build vocabulary
def build_vocab(sentences):
    nltk.download('punkt', quiet=True) # download punkt tokenizer
    tokenised = [[tok.lower() for tok in word_tokenize(
        s, preserve_line=True)] for s in sentences]
    freq = {}
    for sent in tokenised:
        for w in sent:
            freq[w] = freq.get(w, 0) + 1
    most_common = sorted(freq.items(), key=lambda x: x[1], reverse=True)[
        :MAX_VOCAB-2]
    word2idx = {w: i+2 for i, (w, _) in enumerate(most_common)} # start from 2
    word2idx['<PAD>'] = 0 # padding token
    word2idx['<UNK>'] = 1 # unknown token
    return tokenised, word2idx


# embedding matrix from fasttext
def get_embedding_matrix(word2idx):
    vocab_size = len(word2idx) 
    if os.path.exists(EMBEDDING_CACHE): # load from cache if exists (to speed up repeated runs)
        emb = np.load(EMBEDDING_CACHE) 
        if emb.shape[0] == vocab_size: # check if vocab size matches
            return emb
    ft = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False) # load fasttext embeddings
    emb_dim = ft.vector_size
    mat = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for w, idx in word2idx.items():
        mat[idx] = ft[w] if w in ft else np.random.normal(
            scale=0.6, size=(emb_dim,))
    np.save(EMBEDDING_CACHE, mat) # save to cache
    return mat

# encode sequences to indices
def encode_sequences(tokenised, word2idx):
    def encode(tokens):
        idxs = [word2idx.get(t, word2idx['<UNK>']) for t in tokens] # unknown token
        padded = idxs[:MAX_LEN] + [word2idx['<PAD>']] * \
            max(0, MAX_LEN - len(idxs))
        return padded
    return np.array([encode(s) for s in tokenised]) # encode sentences to indices

# dataloader for batching
def make_dataloader(X, y, shuffle=False):
    X_tensor = torch.LongTensor(X) # convert to tensor
    y_tensor = torch.FloatTensor(y) # convert to tensor (float, otherwise BCELoss will fail)
    ds = TensorDataset(X_tensor, y_tensor) # create dataset
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle) # create dataloader


# model definition
class FeedForwardTextClassifier(nn.Module):
    def __init__(self, emb_matrix):
        super().__init__()
        emb_dim = emb_matrix.shape[1] # embedding dimension
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix), freeze=False, padding_idx=0) # freeze padding idx
        layers = []
        input_dim = emb_dim # input dimension
        for h in FNN_HIDDENS:
            layers += [nn.Linear(input_dim, h), nn.ReLU(),
                       nn.Dropout(EMB_DROPOUT)] # hidden layers
            input_dim = h
        layers.append(nn.Linear(input_dim, NUM_CLASSES)) # output layer
        self.net = nn.Sequential(*layers) # create sequential model

    # forward pass
    def forward(self, x):
        emb = self.embedding(x) # embedding lookup
        mask = (x != 0).unsqueeze(-1) # mask for padding
        emb = emb * mask # apply mask
        summed = emb.sum(dim=1) # sum embeddings
        lengths = mask.sum(dim=1).clamp(min=1) # clamp to avoid division by zero
        mean_emb = summed / lengths # mean pooling
        return torch.sigmoid(self.net(mean_emb)) # apply sigmoid to output

def main():
    # load and split data
    X, labels = load_data() # load data
    mlb = MultiLabelBinarizer(classes=list(label_map.keys())) # binarise labels
    y = mlb.fit_transform(labels) 
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y) # split into train and test
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=RANDOM_SEED, stratify=y_tmp) # split into train and validation

    # vocab & embeddings
    tok_train, word2idx = build_vocab(X_train)
    emb_mat = get_embedding_matrix(word2idx)

    # encode sequences
    X_tr_seq = encode_sequences(tok_train, word2idx)
    X_val_tok = [[tok.lower() for tok in word_tokenize(
        s, preserve_line=True)] for s in X_val]
    X_val_seq = encode_sequences(X_val_tok, word2idx)
    X_test_tok = [[tok.lower() for tok in word_tokenize(
        s, preserve_line=True)] for s in X_test]
    X_te_seq = encode_sequences(X_test_tok, word2idx)

    # dataloaders
    loader_train = make_dataloader(X_tr_seq, y_train, shuffle=True)
    loader_val = make_dataloader(X_val_seq, y_val)
    loader_test = make_dataloader(X_te_seq, y_test)

    # model, loss, optimizer
    model = FeedForwardTextClassifier(emb_mat).to(device)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1.0 
    no_improve = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS+1): # training loop
        model.train()
        total_loss = 0
        all_preds, all_targs = [], []
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad() # zero gradients
            preds = model(xb) # forward pass
            loss = criterion(preds, yb) # compute loss
            loss.backward() # backward pass
            optimiser.step()
            total_loss += loss.item() * xb.size(0) # accumulate loss
            all_preds.append((preds >= 0.5).cpu().numpy()) # threshold predictions
            all_targs.append(yb.cpu().numpy()) # append targets
        train_loss = total_loss / len(loader_train.dataset) # average loss
        train_f1 = f1_score(np.vstack(all_targs),
                            np.vstack(all_preds), average='micro') # compute F1 score

        # validation
        model.eval()
        val_loss = 0
        val_preds, val_targs = [], []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device) # forward pass
                preds = model(xb) # forward pass
                val_loss += criterion(preds, yb).item() * xb.size(0) # accumulate loss
                val_preds.append((preds >= 0.5).cpu().numpy()) # threshold predictions
                val_targs.append(yb.cpu().numpy()) # append targets
        val_loss /= len(loader_val.dataset) # average loss
        val_f1 = f1_score(np.vstack(val_targs),
                          np.vstack(val_preds), average='micro') # compute F1 score

        logging.info(f"Epoch {epoch}: Train L={train_loss:.4f}, F1={train_f1:.4f}; "
                     f"Val L={val_loss:.4f}, F1={val_f1:.4f}") # log training and validation loss and F1 score

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

    # load best and test
    model.load_state_dict(best_state)
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in loader_test:
            xb = xb.to(device)
            all_p.append(model(xb).cpu().numpy())
            all_t.append(yb.numpy())
    y_pred_bin = (np.vstack(all_p) >= 0.5).astype(int)
    y_true = np.vstack(all_t)

    # metrics
    micro = f1_score(y_true, y_pred_bin, average='micro') # compute micro F1 score
    macro = f1_score(y_true, y_pred_bin, average='macro') # compute macro F1 score
    logging.info(f"Test Micro-F1: {micro:.4f}, Macro-F1: {macro:.4f}")
    print(f"Test Micro-F1: {micro:.4f}, Macro-F1: {macro:.4f}")


if __name__ == '__main__':
    main()