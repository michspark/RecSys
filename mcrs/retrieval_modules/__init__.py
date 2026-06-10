from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .bge import BGE_MODEL
from .retriever import CLAP_MODEL, HYBRID_MODEL, RERANKER, ANCHOR_CF_MODEL, ANCHOR_BGE_BM25_CF_MODEL


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
    elif retrieval_type == "anchor_cf":
        # BGE 누적쿼리 + anchor 메타 + cf-bpr(beta) 융합
        bge_model = BGE_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        return ANCHOR_CF_MODEL(
            bge_model=bge_model,
            cf_cache_dir=kwargs.get("cf_cache_dir", "./precomputed/reranker"),
            beta=kwargs.get("beta", 0.2),
            alpha_start=kwargs.get("alpha_start", 0.25),
            alpha_step=kwargs.get("alpha_step", 0.05),
            alpha_cap=kwargs.get("alpha_cap", 0.60),
        )
    elif retrieval_type == "anchor_bge_bm25_cf":
        # anchor_cf(best) + BM25 sparse 채널을 RRF 순위융합으로 결합
        bge_model = BGE_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        bm25_model = BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
        return ANCHOR_BGE_BM25_CF_MODEL(
            bge_model=bge_model,
            bm25_model=bm25_model,
            cf_cache_dir=kwargs.get("cf_cache_dir", "./precomputed/reranker"),
            beta=kwargs.get("beta", 0.2),
            alpha_start=kwargs.get("alpha_start", 0.25),
            alpha_step=kwargs.get("alpha_step", 0.05),
            alpha_cap=kwargs.get("alpha_cap", 0.60),
            bm25_topk=kwargs.get("bm25_topk", 150),
            dense_pool=kwargs.get("dense_pool", 200),
            rrf_k=kwargs.get("rrf_k", 60),               # 기본 60(이전 상태). k=10 쓰려면 명시 전달
            rerank_weights=kwargs.get("rerank_weights"),  # 기본 off(None→{}). 켜려면 가중치 명시 전달
        )
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
        reranker = RERANKER(
            track_embedding_dataset=kwargs.get(
                "clap_embedding_dataset",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            user_embedding_dataset=kwargs.get(
                "user_embedding_dataset",
                "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            ),
            split_types=track_split_types,
            cache_dir="./precomputed/reranker",
        )
        return HYBRID_MODEL(
            bge_model=bge_model,
            clap_model=clap_model,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
            keyword_cache_path=kwargs.get("keyword_cache_path"),
            reranker=reranker,
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
