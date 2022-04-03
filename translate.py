from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm.auto import tqdm
from detoxify import Detoxify
tqdm.pandas()


device = torch.device('cuda')

df = pd.read_csv("kriminal_toxic_eng.csv")
# naznaceniya = pd.read_csv("content/naznaceniya_toxic.csv")
# obshhestvo = pd.read_csv("content/obshhestvo_toxic.csv")
# proissestviya = pd.read_csv("content/proissestviya_toxic.csv")
# sport = pd.read_csv("content/sport_toxic.csv")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model.eval()
model.to(device)


def translate(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=300, truncation=True).to(device)
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

model = Detoxify('original')

def toxicity(text):
    results = model.predict(text)['toxicity']
    return results



df['eng_title'] = df.progress_apply(lambda row: translate(row['title']), axis=1)
df['eng_comments'] = df.progress_apply(lambda row: translate(row['comments']), axis=1)
df_women = df.loc[(df['masc'] == 0) & (df['fem'] > 0)]
df_men = df.loc[(df['masc'] > 0) & (df['fem'] == 0)]


df_women['toxicity'] = df_women.progress_apply(lambda row: toxicity(row['eng_comments']), axis=1)
df_men['toxicity'] = df_men.progress_apply(lambda row: toxicity(row['eng_comments']), axis=1)
print(f"women tox: {df_women['toxicity'].mean()}")
print(f"men tox: {df_men['toxicity'].mean()}")

df.to_csv('kriminal_toxic_eng.csv')