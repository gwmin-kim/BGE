import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('bge-reranker-v2-m3')
model.eval()

pairs = [
    ['what is panda?', 'Today is a sunny day'], 
    ['what is panda?', 'The tiger (Panthera tigris) is a member of the genus Panthera and the largest living cat species native to Asia.'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    ]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)