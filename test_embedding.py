from transformers import AutoModel, AutoTokenizer
from typing import List
import torch

class CustomEmbeddingModel:
    def __init__(self, model_name: str, embed_batch_size: int = 32):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_batch_size = embed_batch_size

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        last_hidden_states = outputs.last_hidden_state[:,0]
        norm_last_hidden_states = torch.nn.functional.normalize(last_hidden_states, p=2.0, dim=1)
        embeddings = norm_last_hidden_states.tolist()[0]

        return embeddings

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        cur_batch = []
        result_embeddings = []

        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                embeddings = self._get_text_embeddings(cur_batch)
                result_embeddings.extend(embeddings)
                cur_batch = []

        return result_embeddings

# Example usage
model_name = './models/bge-m3'
texts = "Hello world"
embedding_model = CustomEmbeddingModel(model_name)
embeddings = embedding_model.get_text_embedding_batch(texts)
print(len(embeddings))
print(embeddings)