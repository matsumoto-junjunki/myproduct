#vector2.py
import numpy as np
import pandas as pd
import torch
import scipy.spatial
from transformers import MLukeTokenizer, LukeModel

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        for batch_idx in range(0, len(sentences), batch_size):
            batch = sentences[batch_idx:batch_idx + batch_size]
            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).to("cpu")
            all_embeddings.extend(sentence_embeddings)
        return torch.stack(all_embeddings)

# 検索ロジックを関数化
def search_program(query, top_n=3):
    # モデルとデータの初期化
    MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
    model = SentenceLukeJapanese(MODEL_NAME)

    csv_file_path = "program_re.csv"
    data = pd.read_csv(csv_file_path, index_col=0)
    sentences = data["content"].tolist()
    precomputed_embeddings = np.load("sentence_embeddings.npy")

    # キーワード一致検索
    key_matches = data[data["content"].str.contains(query)]
    key_titles = key_matches["title"].tolist()

    # クエリをベクトル化
    query_embedding = model.encode([query], batch_size=1)

    # 類似度計算
    distances = scipy.spatial.distance.cdist(
        query_embedding, precomputed_embeddings, metric="cosine"
    )[0]

    # 類似度上位N件を取得
    results = sorted(zip(range(len(distances)), distances), key=lambda x: x[1])[:top_n]

    recommended_programs = [
        {
            "title": data.iloc[idx, 1],
            "content": sentences[idx].strip(),
            "distance": distance,
        }
        for idx, distance in results
    ]

    return {"key_matches": key_titles, "recommendations": recommended_programs}
