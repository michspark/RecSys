import os
import json
from typing import List
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForCausalLM
from .extract_query_comp import extract_keyword_query


class CLAP_MODEL:
    """Text-to-audio retriever using pre-computed track audio embeddings.

    Loads the CLAP text encoder from HuggingFace and aligns it against
    pre-computed audio embeddings stored in `talkpl-ai/TalkPlayData-2-Track-Embeddings`.
    Audio embeddings are cached locally to avoid re-downloading.
    """

    def __init__(self, model_name: str = "laion/larger_clap_music",
        embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list[str] = ["all_tracks"],
        cache_dir: str = "./cache", device: str | None = None,) -> None:
        self.model_name = model_name
        self.embedding_dataset = embedding_dataset
        self.split_types = split_types
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(cache_dir, "clap")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.embeddings, self.track_ids = self._load_or_build_index()
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

    def _load_or_build_index(self):
        emb_path = os.path.join(self.index_dir, "embeddings.pt")
        ids_path = os.path.join(self.index_dir, "track_ids.json")

        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embeddings = torch.load(emb_path, map_location="cpu")
            with open(ids_path) as f:
                track_ids = json.load(f)
            return embeddings, track_ids

        dataset = load_dataset(self.embedding_dataset)
        splits = [dataset[s] for s in self.split_types if s in dataset]
        full_dataset = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        track_ids_raw = list(full_dataset["track_id"])
        raw_embeddings = list(full_dataset["audio-laion_clap"])  # list of 512-dim float lists

        # Filter out entries with missing/empty embeddings
        valid = [(tid, emb) for tid, emb in zip(track_ids_raw, raw_embeddings) if emb and len(emb) == 512]
        track_ids, raw_embeddings = zip(*valid)
        track_ids = list(track_ids)
        print(f"CLAP: loaded {len(track_ids)} tracks from split(s) {self.split_types}")
        embeddings = torch.tensor(raw_embeddings, dtype=torch.float32)
        embeddings = F.normalize(embeddings, p=2, dim=1) # normalize for cosine similarity

        os.makedirs(self.index_dir, exist_ok=True)
        torch.save(embeddings, emb_path)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)

        return embeddings, track_ids

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query into a normalized CLAP text embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.to(self.device) for k, v in inputs.items()
                       if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            text_outputs = self.model.text_model(**text_inputs)
            pooled = text_outputs.pooler_output          # [1, hidden_dim]
            text_emb = self.model.text_projection(pooled)  # [1, 512] This maps to the same space as audio embeddings
            text_emb = F.normalize(text_emb, p=2, dim=1)
        return text_emb.squeeze(0).cpu()  # [512]

    def text_to_audio_retrieval(self, mood_text: str, topk: int = 20) -> List[str]:
        """Retrieve top-k track IDs by cosine similarity between text and audio embeddings."""
        query_emb = self.encode_text(mood_text)
        scores = torch.matmul(self.embeddings, query_emb)  # [N]
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]


class HYBRID_MODEL:
    """Hybrid retriever combining BGE text scores with CLAP audio scores.

    BGE retrieves based on metadata text similarity; CLAP retrieves based on
    audio-semantic similarity from keywords extracted by Qwen3-4B.
    Final score = alpha * normalize(bge_score) + beta * normalize(clap_score).

    The interface matches BGE_MODEL so it is a drop-in for `crs_baseline.py`.
    """

    def __init__(self, bge_model, clap_model: CLAP_MODEL,
        alpha: float = 0.6, beta: float = 0.4, keyword_model_name: str = "Qwen/Qwen3-4B",) -> None:
        self.bge_model = bge_model
        self.clap_model = clap_model
        self.alpha = alpha
        self.beta = beta
        self.track_ids = bge_model.track_ids  # authoritative track order

        # Load Qwen3-4B for keyword extraction
        self.keyword_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keyword_tokenizer, self.keyword_model = self._load_keyword_model(keyword_model_name)

        # Pre-align CLAP embeddings to BGE's track order.
        # Tracks missing from CLAP get a zero vector (CLAP contributes nothing).
        clap_dim = clap_model.embeddings.shape[1]
        aligned = torch.zeros(len(self.track_ids), clap_dim)
        for i, tid in enumerate(self.track_ids):
            if tid in clap_model.track_id_to_idx:
                j = clap_model.track_id_to_idx[tid]
                aligned[i] = clap_model.embeddings[j]
        self.clap_aligned = aligned  # [N_bge, clap_dim]

    def _load_keyword_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        model.to(self.keyword_device).eval()
        return tokenizer, model

    def _bge_scores(self, query: str) -> torch.Tensor:
        """Compute cosine similarity scores between query and all BGE track embeddings."""
        instruction_query = self.bge_model.QUERY_INSTRUCTION + query
        self.bge_model.model.eval()
        with torch.no_grad():
            batch = self.bge_model.tokenizer(
                [instruction_query],
                padding=True,
                truncation=True,
                max_length=self.bge_model.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.bge_model.device) for k, v in batch.items()}
            outputs = self.bge_model.model(**batch)
            query_emb = outputs.last_hidden_state[:, 0]
            query_emb = F.normalize(query_emb, p=2, dim=1).cpu().squeeze(0)
        return torch.matmul(self.bge_model.embeddings, query_emb)  # [N]

    def _clap_scores(self, mood_text: str) -> torch.Tensor:
        """Compute cosine similarity scores between keywords and all CLAP audio embeddings."""
        query_emb = self.clap_model.encode_text(mood_text)  # [dim]
        return torch.matmul(self.clap_aligned, query_emb)   # [N]

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-9:
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        """Retrieve top-k tracks by combining BGE and CLAP scores."""
        lm_components = (self.keyword_model, self.keyword_tokenizer, self.keyword_device)
        mood_text = extract_keyword_query(query, lm_components)
        bge_scores = self._normalize(self._bge_scores(query))
        clap_scores = self._normalize(self._clap_scores(mood_text))
        combined = self.alpha * bge_scores + self.beta * clap_scores
        topk = min(topk, combined.shape[0])
        top_indices = torch.topk(combined, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        """Retrieve top-k tracks for multiple queries."""
        return [self.text_to_item_retrieval(q, topk) for q in queries]
