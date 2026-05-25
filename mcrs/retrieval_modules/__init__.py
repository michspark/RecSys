from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .bge import BGE_MODEL
from .retriever import CLAP_MODEL, HYBRID_MODEL


def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        **kwargs
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bge":
        return BGE_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "hybrid":
        bge_model = BGE_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        clap_model = CLAP_MODEL(
            cache_dir=cache_dir,
            embedding_dataset=kwargs.get(
                "clap_embedding_dataset",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            split_types=track_split_types,
        )
        return HYBRID_MODEL(
            bge_model=bge_model,
            clap_model=clap_model,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
