import os
import json
from typing import List
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModel, AutoProcessor


class SIGLIP_MODEL:
    """Text-to-image retriever using pre-computed track album-cover embeddings.

    Loads the SigLIP2 text encoder from HuggingFace and aligns it against
    pre-computed album-cover image embeddings stored in the column
    `image-siglip2` of `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`.
    SigLIP2 is a contrastive image-text model, so its text and image
    embeddings live in the SAME 768-dim space — a text query can be matched
    against album-cover image embeddings by cosine similarity.
    Image embeddings are cached locally to avoid re-downloading.
    """

    # Image embeddings are 768-dim (SigLIP2 "base" family projection dim).
    EMBED_DIM: int = 768

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224",
        embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list[str] = ["all_tracks", "test_tracks"],
        cache_dir: str = "./cache", device: str | None = None,) -> None:
        self.model_name = model_name
        self.embedding_dataset = embedding_dataset
        self.split_types = split_types
        self.cache_dir = cache_dir
        # Cache dir keyed by sorted split names, mirroring CLAP_MODEL's pattern.
        split_tag = "_".join(sorted(split_types))
        self.index_dir = os.path.join(cache_dir, f"siglip_{split_tag}")

        # Pick GPU if available, else CPU.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # SigLIP2 is loaded via the generic Auto classes (it is a Siglip2Model).
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Pre-computed image embeddings + their track-id order (cosine-normalized).
        self.embeddings, self.track_ids = self._load_or_build_index()
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

    def _load_or_build_index(self):
        emb_path = os.path.join(self.index_dir, "embeddings.pt")
        ids_path = os.path.join(self.index_dir, "track_ids.json")

        # Fast path: reuse the cached index if it exists.
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            embeddings = torch.load(emb_path, map_location="cpu")
            with open(ids_path) as f:
                track_ids = json.load(f)
            return embeddings, track_ids

        # Build path: pull the precomputed image embeddings from the HF dataset.
        dataset = load_dataset(self.embedding_dataset)
        splits = [dataset[s] for s in self.split_types if s in dataset]
        full_dataset = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

        track_ids_raw = list(full_dataset["track_id"])
        raw_embeddings = list(full_dataset["image-siglip2"])  # list of 768-dim float lists

        # Drop tracks with missing/empty/wrong-dim image embeddings.
        valid = [(tid, emb) for tid, emb in zip(track_ids_raw, raw_embeddings)
                 if emb and len(emb) == self.EMBED_DIM]
        track_ids, raw_embeddings = zip(*valid)
        track_ids = list(track_ids)
        print(f"SigLIP: loaded {len(track_ids)} tracks from split(s) {self.split_types}")
        embeddings = torch.tensor(raw_embeddings, dtype=torch.float32)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize for cosine similarity

        # Persist the index so subsequent runs skip the download/build.
        os.makedirs(self.index_dir, exist_ok=True)
        torch.save(embeddings, emb_path)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)

        return embeddings, track_ids

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text query into a normalized SigLIP2 text embedding [768]."""
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Batch-encode text queries into normalized SigLIP2 text embeddings [B, 768].

        SigLIP2 pads every sequence to a fixed length (64) — `padding="max_length"`
        is the convention the model was trained with, not dynamic padding.
        """
        all_embs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt",
                                        padding="max_length", max_length=64, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # get_text_features runs the text tower + pooling head → shared space.
                # transformers 5.x returns a model-output object; the final text
                # embedding is its `pooler_output` ([B, 768]).
                out = self.model.get_text_features(**inputs)
                text_emb = out.pooler_output if hasattr(out, "pooler_output") else out
                text_emb = F.normalize(text_emb, p=2, dim=1)
                all_embs.append(text_emb.cpu())
        return torch.cat(all_embs, dim=0)

    def text_to_image_scores(self, text: str) -> torch.Tensor:
        """Cosine similarity between a text query and ALL tracks' image embeddings [N]."""
        query_emb = self.encode_text(text)
        return torch.matmul(self.embeddings, query_emb)  # [N]

    def text_to_image_retrieval(self, text: str, topk: int = 20) -> List[str]:
        """Retrieve top-k track IDs by text→image cosine similarity."""
        scores = self.text_to_image_scores(text)
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]

    def get_image_embedding(self, track_id: str) -> "torch.Tensor | None":
        """Return the pre-computed (L2-normalized) image embedding for a track ID.
        Returns None if the track has no image embedding in the index."""
        idx = self.track_id_to_idx.get(track_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def image_to_image_scores(self, anchor_track_id: str) -> "torch.Tensor | None":
        """Cosine similarity between the anchor track's cover and ALL tracks' covers.
        Returns a [N] tensor aligned with self.track_ids order,
        or None if the anchor track has no image embedding."""
        anchor_emb = self.get_image_embedding(anchor_track_id)
        if anchor_emb is None:
            return None
        scores = torch.matmul(self.embeddings, anchor_emb)  # [N]
        return scores

    def image_to_image_retrieval(self, anchor_track_id: str, topk: int = 20,
                                 exclude_self: bool = True) -> List[str]:
        """Retrieve top-k track IDs most similar in album-cover image to the anchor."""
        scores = self.image_to_image_scores(anchor_track_id)
        if scores is None:
            return []
        scores = scores.clone()

        if exclude_self:
            idx = self.track_id_to_idx.get(anchor_track_id)
            if idx is not None:
                scores[idx] = -1e9  # effectively exclude self from top-k
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]
