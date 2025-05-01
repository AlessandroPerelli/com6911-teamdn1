import os
import random
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
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report

from transformers import AutoModel, AutoTokenizer

DATA_PATH = r'C:/Users/molly/COM6911/com6911-teamdn1/annotated_data/full/combined_complete.xlsx'
FASTTEXT_PATH = r'C:/Users/molly/COM6911/cc.en.300.vec/cc.en.300.vec'
RANDOM_SEED = 42
MODEL_TYPE    = 'cnn_rnn'  # choose from ['textcnn','bilstm','cnn_rnn','transformer']
# similar f1 scores
#2025-05-01 17:12:29,548 INFO Test micro-F1: 0.6876
#2025-05-01 17:12:29,549 INFO Test macro-F1: 0.6735

#bilstm
#2025-05-01 17:13:21,862 INFO Test micro-F1: 0.6878
#2025-05-01 17:13:21,862 INFO Test macro-F1: 0.6748

#cnn_rnn
#2025-05-01 17:15:25,367 INFO Test micro-F1: 0.6839
#2025-05-01 17:15:25,367 INFO Test macro-F1: 0.6749

MAX_VOCAB = 1000
MAX_LEN = 100
EMBEDDING_DIM = 300
FILTER_SIZES = [3,4,5]
NUM_FILTERS = 100
DROPOUT = 0.5
LR = 0.001
BATCH_SIZE = 32
NUM_CLASSES = 4

MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

LABEL_MAP = {1:'mobility',2:'self_care_domestic_life',3:'interpersonal_interactions',4:'communication_cognition'}

logging.info('Loading data from Excel...')
df = pd.read_excel(DATA_PATH)

def parse_labels(entry):
    parts = str(entry).split(',')
    return [int(p.strip().strip('"').strip("'")) for p in parts if p.strip().strip('"').strip("'").isdigit() and int(p.strip().strip('"').strip("'")) in LABEL_MAP]

df['labels_list'] = df['label'].apply(parse_labels)

pos_df = df[df['labels_list'].map(len)>0]
neg_df = df[df['labels_list'].map(len)==0]
neg_frac = 0.05
neg_sample = neg_df.sample(frac=neg_frac, random_state=RANDOM_SEED)
df = pd.concat([pos_df,neg_sample]).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
logging.info(f"After under-sampling: {len(pos_df)} positives + {len(neg_sample)} negatives = {len(df)} total")

mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
y = mlb.fit_transform(df['labels_list'])
X_sent = df['sentence'].astype(str).tolist()

X_tmp,X_test,y_tmp,y_test = train_test_split(X_sent,y,test_size=0.2,random_state=RANDOM_SEED,stratify=y)
X_train,X_val,y_train,y_val = train_test_split(X_tmp,y_tmp,test_size=0.25,random_state=RANDOM_SEED,stratify=y_tmp)
logging.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

def tokenize(text):
    return [tok.lower() for tok in word_tokenize(text, preserve_line=True)]

X_train_tok = [tokenize(s) for s in X_train]
X_val_tok = [tokenize(s) for s in X_val]
X_test_tok = [tokenize(s) for s in X_test]

counter={}
for sent in X_train_tok:
    for tok in sent:
        counter[tok]=counter.get(tok,0)+1
most_common=sorted(counter.items(),key=lambda x:x[1],reverse=True)[:MAX_VOCAB-2]
word2idx={w:i+2 for i,(w,_) in enumerate(most_common)}
word2idx['<PAD>']=0
word2idx['<UNK>']=1
vocab_size=len(word2idx)

CACHE_PATH='embedding_matrix.npy'
if os.path.exists(CACHE_PATH):
    embedding_matrix=np.load(CACHE_PATH)
else:
    ft=KeyedVectors.load_word2vec_format(FASTTEXT_PATH,binary=False)
    EMBEDDING_DIM=ft.vector_size
    embedding_matrix=np.zeros((vocab_size,EMBEDDING_DIM),dtype=np.float32)
    for w,idx in word2idx.items():
        embedding_matrix[idx]=ft[w] if w in ft else np.random.normal(scale=0.6,size=(EMBEDDING_DIM,))
    np.save(CACHE_PATH,embedding_matrix)

def encode(tokens):
    idxs=[word2idx.get(t,word2idx['<UNK>']) for t in tokens]
    if len(idxs)<MAX_LEN: idxs+=[word2idx['<PAD>']]*(MAX_LEN-len(idxs))
    else: idxs=idxs[:MAX_LEN]
    return idxs

X_tr_seq=np.array([encode(s) for s in X_train_tok])
X_val_seq=np.array([encode(s) for s in X_val_tok])
X_te_seq=np.array([encode(s) for s in X_test_tok])

class TextCNN(nn.Module):
    def __init__(self,vocab_size,emb_dim,num_classes,filter_sizes,num_filters,dropout):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix),freeze=False,padding_idx=word2idx['<PAD>'])
        self.convs1=nn.ModuleList([nn.Conv2d(1,num_filters,(fs,emb_dim)) for fs in filter_sizes])
        self.bns1=nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in filter_sizes])
        self.convs2=nn.ModuleList([nn.Conv2d(num_filters,num_filters,(3,1),padding=(1,0)) for _ in filter_sizes])
        self.bns2=nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in filter_sizes])
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(num_filters*len(filter_sizes),num_classes)
    def forward(self,x):
        x=self.embedding(x).unsqueeze(1)
        pooled=[]
        for c1,b1,c2,b2 in zip(self.convs1,self.bns1,self.convs2,self.bns2):
            y=F.relu(b1(c1(x)))
            y=F.relu(b2(c2(y))).squeeze(3)
            pooled.append(y.max(dim=2)[0])
        return torch.sigmoid(self.fc(self.dropout(torch.cat(pooled,dim=1))))

class BiLSTMAttention(nn.Module):
    def __init__(self,vocab_size,emb_dim,num_classes,hidden_dim=128,dropout=0.5):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix),freeze=False,padding_idx=word2idx['<PAD>'])
        self.lstm=nn.LSTM(emb_dim,hidden_dim,bidirectional=True,batch_first=True)
        self.attn=nn.Linear(hidden_dim*2,1)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(hidden_dim*2,num_classes)
    def forward(self,x):
        emb,_=self.lstm(self.embedding(x))
        weights=torch.softmax(self.attn(emb).squeeze(2),dim=1)
        context=torch.bmm(weights.unsqueeze(1),emb).squeeze(1)
        return torch.sigmoid(self.fc(self.dropout(context)))

class CNNRNNHybrid(nn.Module):
    def __init__(self,vocab_size,emb_dim,num_classes,filter_sizes,num_filters,rnn_hidden,dropout):
        super().__init__()
        self.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix),freeze=False,padding_idx=word2idx['<PAD>'])
        self.convs=nn.ModuleList([nn.Conv1d(emb_dim,num_filters,fs) for fs in filter_sizes])
        self.gru=nn.GRU(num_filters*len(filter_sizes),rnn_hidden,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(rnn_hidden,num_classes)
    def forward(self,x):
        emb=self.embedding(x).transpose(1,2)
        convs=[F.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        cat=torch.cat(convs,dim=1).unsqueeze(1)
        _,h=self.gru(cat)
        return torch.sigmoid(self.fc(self.dropout(h.squeeze(0))))

# (Experimental)
class TransformerClassifier(nn.Module):
    def __init__(self,model_name,num_classes):
        super().__init__()
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.transformer=AutoModel.from_pretrained(model_name)
        hidden_size=self.transformer.config.hidden_size
        self.fc=nn.Linear(hidden_size,num_classes)
        self.dropout=nn.Dropout(0.3)
    def forward(self,x):
        enc=self.tokenizer(x,padding=True,truncation=True,max_length=MAX_LEN,return_tensors='pt')
        enc={k:v.to(device) for k,v in enc.items()}
        out=self.transformer(**enc).pooler_output
        return torch.sigmoid(self.fc(self.dropout(out)))

model=None
if MODEL_TYPE=='textcnn': model=TextCNN(vocab_size,EMBEDDING_DIM,NUM_CLASSES,FILTER_SIZES,NUM_FILTERS,DROPOUT)
elif MODEL_TYPE=='bilstm': model=BiLSTMAttention(vocab_size,EMBEDDING_DIM,NUM_CLASSES,hidden_dim=128,dropout=DROPOUT)
elif MODEL_TYPE=='cnn_rnn': model=CNNRNNHybrid(vocab_size,EMBEDDING_DIM,NUM_CLASSES,FILTER_SIZES,NUM_FILTERS,rnn_hidden=128,dropout=DROPOUT)
elif MODEL_TYPE=='han': model=HAN(vocab_size,EMBEDDING_DIM,NUM_CLASSES,word_hidden=64,sent_hidden=64,dropout=DROPOUT)
elif MODEL_TYPE=='transformer': model=TransformerClassifier('emilyalsentzer/Bio_ClinicalBERT',NUM_CLASSES)
else: raise ValueError(f"Unknown MODEL_TYPE '{MODEL_TYPE}'")
model=model.to(device)
logging.info(f"Using model type: {MODEL_TYPE}")

if MODEL_TYPE=='transformer':
    train_ds=TensorDataset(torch.arange(len(X_train)))
    val_ds=TensorDataset(torch.arange(len(X_val)))
    test_ds=TensorDataset(torch.arange(len(X_test)))
else:
    train_ds=TensorDataset(torch.LongTensor(X_tr_seq),torch.FloatTensor(y_train))
    val_ds=TensorDataset(torch.LongTensor(X_val_seq),torch.FloatTensor(y_val))
    test_ds=TensorDataset(torch.LongTensor(X_te_seq),torch.FloatTensor(y_test))
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE)
test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE)

criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=LR)

best_val_f1=0.0
epochs_no_improve=0
best_state=None
for epoch in range(1,MAX_EPOCHS+1):
    model.train()
    total_loss=0.0
    for batch in train_loader:
        if MODEL_TYPE=='transformer': idxs=batch[0].tolist();Xb=[X_train[i]for i in idxs];yb=torch.FloatTensor(y_train[idxs]).to(device)
        else: Xb,yb=batch[0].to(device),batch[1].to(device)
        optimizer.zero_grad();preds=model(Xb);loss=criterion(preds,yb);loss.backward();optimizer.step()
        total_loss+=loss.item()*(yb.size(0)if MODEL_TYPE!='transformer'else len(idxs))
    model.eval()
    all_p,all_t=[],[]
    with torch.no_grad():
        for batch in val_loader:
            if MODEL_TYPE=='transformer': idxs=batch[0].tolist();Xb=[X_val[i]for i in idxs];yb=torch.FloatTensor(y_val[idxs]).to(device)
            else: Xb,yb=batch[0].to(device),batch[1].to(device)
            p=(model(Xb)>=0.5).float().cpu().numpy();all_p.append(p);all_t.append(yb.cpu().numpy())
    all_p=np.vstack(all_p);all_t=np.vstack(all_t)
    val_f1=f1_score(all_t,all_p,average='micro')
    logging.info(f"Epoch {epoch}: Val micro-F1 = {val_f1:.4f}")
    if val_f1>best_val_f1: best_val_f1=val_f1;epochs_no_improve=0;best_state=model.state_dict()
    else: epochs_no_improve+=1
    if epochs_no_improve>=EARLY_STOPPING_PATIENCE: logging.info("Early stopping triggered");break
model.load_state_dict(best_state)
model.eval()
all_p,all_t=[],[]
with torch.no_grad():
    for batch in test_loader:
        if MODEL_TYPE=='transformer': idxs=batch[0].tolist();Xb=[X_test[i]for i in idxs];yb=torch.FloatTensor(y_test[idxs]).to(device)
        else: Xb,yb=batch[0].to(device),batch[1].to(device)
        p=(model(Xb)>=0.5).float().cpu().numpy();all_p.append(p);all_t.append(yb.cpu().numpy())
all_p=np.vstack(all_p);all_t=np.vstack(all_t)
test_micro=f1_score(all_t,all_p,average='micro')
test_macro=f1_score(all_t,all_p,average='macro')
logging.info(f"Test micro-F1: {test_micro:.4f}")
logging.info(f"Test macro-F1: {test_macro:.4f}")
cm=multilabel_confusion_matrix(all_t,all_p)
class_names=list(LABEL_MAP.values())
for i,cls in enumerate(class_names):tn,fp,fn,tp=cm[i].ravel();logging.info(f"Confusion matrix for {cls}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
logging.info("Classification Report:\n"+classification_report(all_t,all_p,target_names=class_names))
