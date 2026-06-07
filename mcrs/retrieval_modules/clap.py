import os
import json
from typing import List
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import ClapModel, ClapProcessor

class CLAP_MODEL:
    """Text-to-audio retriever using pre-computed track audio embeddings.

    Loads the CLAP text encoder from HuggingFace and aligns it against
    pre-computed audio embeddings stored in `talkpl-ai/TalkPlayData-2-Track-Embeddings`.
    Audio embeddings are cached locally to avoid re-downloading.
    """

    def __init__(self, model_name: str = "laion/larger_clap_music",
        embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list[str] = ["all_tracks", "test_tracks"],
        cache_dir: str = "./cache", device: str | None = None,) -> None:
        self.model_name = model_name
        self.embedding_dataset = embedding_dataset
        self.split_types = split_types
        self.cache_dir = cache_dir
        split_tag = "_".join(sorted(split_types))
        self.index_dir = os.path.join(cache_dir, f"clap_{split_tag}")

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

    def get_audio_embedding(self, track_id: str) -> "torch.Tensor | None":
        """Return the pre-computed (L2 Normalized) audio embeddinf for a track ID.
        Returns None if the track has no audio embedding in the index"""
        idx = self.track_id_to_idx.get(track_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def audio_to_audio_scores(self, anchor_track_id: str) -> "torch.Tensor | None":
        """ Cosine similarity between the anchor track's audio and ALL tracks' audio.
        Returns a [N] tensor aligned with self.track_ids order,
        or None if the anchor track has no audio embedding
        """
        anchor_emb = self.get_audio_embedding(anchor_track_id)
        if anchor_emb is None:
            return None
        scores = torch.matmul(self.embeddings, anchor_emb)  # [N]
        return scores

    def audio_to_audio_retrieval(self, anchor_track_id: str, topk: int=20,
                                 exclude_self: bool = True) -> List[str]:
        """Retrieve top-k track IDs most similar in audio to the anchor track"""
        scores = self.audio_to_audio_scores(anchor_track_id)
        if scores is None:
            return []
        scores = scores.clone()

        if exclude_self:
            idx = self.track_id_to_idx.get(anchor_track_id)
            if idx is not None:
                scores[idx] = -1e9 # effectively exclude self from top-k
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]
    
