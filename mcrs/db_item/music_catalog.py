import os
import torch
import json
from datasets import load_dataset, concatenate_datasets

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False, include_track_id: bool = True):
        metadata = self.metadata_dict[track_id]
        track_id = metadata['track_id']
        # Optionally prefix the raw track_id. The LLM prompt path passes include_track_id=False so the
        # model never sees the opaque ID (it should only get human-readable fields); other callers
        # (retrieval corpus, keyword preprocessing, analysis) keep the default and still get it.
        entity_str = f"track_id: {track_id}" if include_track_id else ""
        for corpus_type in self.corpus_types:
            corpus_type_value = ", ".join(metadata[corpus_type]).lower()
            # Skip the leading ", " when the track_id prefix was omitted (entity_str still empty).
            separator = ", " if entity_str else ""
            entity_str += f"{separator}{corpus_type}: {corpus_type_value}"
        return entity_str
    
# Example Return "track_id: 12345, artist_name: daft punk, pharrell williams, album_name: random access memories".