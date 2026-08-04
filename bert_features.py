import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# ==========================================
# 1. تحميل البيانات المُنظفة
# ==========================================
df = pd.read_csv('data/preprocessed.csv')
df = df.dropna(subset=['clean_text'])


texts = df['text'].astype(str).tolist()


print("جاري تحميل BERT... (أول مرة بتاخد وقت أطول)")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # وضع التقييم (مش تدريب)


def get_bert_embedding(text, tokenizer, model, max_length=64):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    with torch.no_grad():   
        outputs = model(**inputs)
  
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


print(f"جاري استخراج BERT embeddings لـ {len(texts)} نص... (ممكن ياخد وقت طويل)")

embeddings_list = []
batch_size = 500  

for i, text in enumerate(texts):
    emb = get_bert_embedding(text, tokenizer, model)
    embeddings_list.append(emb)
    if (i + 1) % batch_size == 0:
        print(f"  تمت معالجة {i + 1} / {len(texts)} نص...")

X_emb = np.array(embeddings_list)
print("شكل مصفوفة الـ Embeddings:", X_emb.shape)


np.save('X_embeddings.npy', X_emb)
print("✅ تم حفظ مصفوفة الـ BERT Embeddings بنجاح!")