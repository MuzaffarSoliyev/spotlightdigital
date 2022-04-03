from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm.auto import tqdm
tqdm.pandas()


device = torch.device('cuda')

df = pd.read_csv("proissestviya_toxic.csv")
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


df['eng_title'] = df.progress_apply(lambda row: translate(row['title']), axis=1)
df['eng_comments'] = df.progress_apply(lambda row: translate(row['comments']), axis=1)
df.to_csv('proissestviya_toxic_eng.csv')