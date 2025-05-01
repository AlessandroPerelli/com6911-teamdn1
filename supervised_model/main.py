import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import multilabel_confusion_matrix, classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')

df = pd.read_excel('C:/Users/molly/COM6911/com6911-teamdn1/annotated_data/full/combined_complete.xlsx')
label_mapping = {
    1: 'mobility',
    2: 'self_care_domestic_life',
    3: 'interpersonal_interactions',
    4: 'communication_cognition'
}
def parse_labels(entry):
    parts = str(entry).split(',')
    return [
        int(p.strip().strip('"').strip("'"))
        for p in parts
        if p.strip().strip('"').strip("'").isdigit()
        and int(p.strip().strip('"').strip("'")) in label_mapping
    ]
df['labels_list'] = df['label'].apply(parse_labels)

mlb = MultiLabelBinarizer(classes=list(label_mapping.keys()))
y = mlb.fit_transform(df['labels_list'])
X = df['sentence'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

max_vocab = 20000
max_len   = 100

counter = Counter()
for s in X_train:
    counter.update(s.lower().split())
most_common = counter.most_common(max_vocab-2)
word2idx = {w:i+2 for i,(w,_) in enumerate(most_common)}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1

def encode(s):
    toks = s.lower().split()
    idxs = [word2idx.get(t,1) for t in toks]
    if len(idxs)<max_len:
        idxs += [0]*(max_len-len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

X_tr_seq = np.array([encode(s) for s in X_train])
X_te_seq = np.array([encode(s) for s in X_test])

batch_size = 64
train_ds = TensorDataset(torch.LongTensor(X_tr_seq),
                         torch.FloatTensor(y_train))
test_ds  = TensorDataset(torch.LongTensor(X_te_seq),
                         torch.FloatTensor(y_test))

train_loader     = DataLoader(train_ds, batch_size, shuffle=True)
train_eval_loader= DataLoader(train_ds, batch_size, shuffle=False)
test_loader      = DataLoader(test_ds,  batch_size, shuffle=False)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes=[3,4,5], num_filters=100, drop=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(num_filters*len(filter_sizes), num_classes)
    def forward(self, x):
        x = self.embed(x)     
        x = x.unsqueeze(1)    
        outs = [
            torch.relu(conv(x)).squeeze(3) 
            for conv in self.convs
        ]
        pools = [o.max(dim=2)[0] for o in outs] 
        cat = torch.cat(pools, dim=1)  
        drop = self.dropout(cat)
        return torch.sigmoid(self.fc(drop)) 

vocab_size  = len(word2idx)
embed_dim   = 128
num_classes = y_train.shape[1]

model     = TextCNN(vocab_size, embed_dim, num_classes).to(device)
optimiser = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

n_epochs = 10
for epoch in range(1, n_epochs+1):
    model.train()
    total_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimiser.zero_grad()
        preds = model(Xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()*Xb.size(0)
    avg_loss = total_loss / len(train_ds)

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in train_eval_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            p = (model(Xb) >= 0.5).float().cpu().numpy()
            all_p.append(p); all_t.append(yb.cpu().numpy())
    all_p   = np.vstack(all_p)
    all_t = np.vstack(all_t)

    micro_f1 = f1_score(all_t, all_p, average='micro')
    macro_f1 = f1_score(all_t, all_p, average='macro')

    print(f"Epoch {epoch} — Val micro-F1: {micro_f1:.4f}, macro-F1: {macro_f1:.4f}")

model.eval()
test_preds, test_targs = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        p = (model(Xb) >= 0.5).float().cpu().numpy()
        test_preds.append(p); test_targs.append(yb.cpu().numpy())
test_preds = np.vstack(test_preds)
test_targs = np.vstack(test_targs)
test_acc = (test_preds == test_targs).mean()
test_f1_micro = f1_score(test_targs, test_preds, average='micro')
test_f1_macro = f1_score(test_targs, test_preds, average='macro')

print(f"\nTest micro-F1: {test_f1_micro:.4f}")
print(f"Test macro-F1: {test_f1_macro:.4f}")
#print(f"\nTest Accuracy: {test_acc:.4f}")

cm = multilabel_confusion_matrix(test_targs, test_preds)
class_names = ['mobility', 'self_care_domestic_life',
               'interpersonal_interactions', 'communication_cognition']

for i, cls in enumerate(class_names):
    tn, fp, fn, tp = cm[i].ravel()
    print(f"Confusion matrix for {cls}:")
    print(f"  TP: {tp:5d}   FP: {fp:5d}")
    print(f"  FN: {fn:5d}   TN: {tn:5d}\n")

print(classification_report(test_targs, test_preds, target_names=class_names))