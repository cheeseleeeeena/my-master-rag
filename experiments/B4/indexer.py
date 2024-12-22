# works fine in "cuda12" env on 9104
# works fine on "vllm" env on 9104
# 50 secs for 416 papers.

import json
import re
import time
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set
import utils

# Constants
os.environ["HF_HOME"] = "/workspace/P76125041/.cache/"
RETRIEVER = "sbert"
MODE = "full"
CUDA_DEVICE = 0

# Model paths
retriever_map: Dict[str, str] = {
    "sbert": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "stella": "/workspace/P76125041/models/stella_en_400M_v5"
}


def format_paragraphs(raw_paras: List[dict], mode: str) -> List[str]:
    """Format paragraphs based on the selected mode."""
    if mode == "full":
        return [
            f"Section Title: {para.get('section_name', '[UNNAMED SECTION]')}\n\nText: {para.get('text', '')}"
            for para in raw_paras
        ]
    elif mode == "toptitle":
        return [
            f"Section Title: {re.sub(r':::.*', '', para.get('section_name', '[UNNAMED SECTION]')).strip()}\n\nText: {para.get('text', '')}"
            for para in raw_paras
        ]
    else:
        return [para.get("text", "") for para in raw_paras]


def encode_paragraphs(
    raw_paras: List[dict], mode: str, embedding_model
) -> List[List[float]]:
    """Format and encode paragraphs into embeddings."""
    paragraphs = format_paragraphs(raw_paras, mode)
    para_embeddings = embedding_model.encode(paragraphs)

    # Validate embeddings
    if len(para_embeddings) != len(paragraphs):
        raise ValueError("Mismatch in number of paragraphs and embeddings!")

    return para_embeddings



if __name__ == "__main__":
    start_time = time.time()

    # Load data files
    test_papers = utils.load_json(Path("qasper/test_papers.json"))
    test_questions = utils.load_json(Path("qasper/test_questions.json"))
    
    # initialize retriever
    ### Stella
    if RETRIEVER == "stella":
        embedding_model = SentenceTransformer(retriever_map.get(RETRIEVER), trust_remote_code=True).cuda()
    ### SBERT
    else:
        embedding_model = SentenceTransformer(retriever_map[RETRIEVER])
        embedding_model.to(f"cuda:{CUDA_DEVICE}")

    # Prepare paragraph embeddings
    all_paper_ids: Set[str] = {q["from_paper"] for q in test_questions.values()}
    paper_para_embeddings: Dict[str, List[List[float]]] = {
        paper_id: encode_paragraphs(
            test_papers[paper_id].values(), MODE, embedding_model
        )
        for paper_id in all_paper_ids
    }
    print(f"Finished indexing {len(all_paper_ids)} papers.")

    # Convert numpy arrays inside paper_para_embeddings to lists
    paper_para_embeddings_list: Dict[str, List[List[float]]] = {
        paper_id: [embedding.tolist() for embedding in embeddings]
        for paper_id, embeddings in paper_para_embeddings.items()
    }

    # Save embeddings
    embedding_file = f"qasper/test_paper_embeddings_{RETRIEVER}_{MODE}.json"
    with open(embedding_file, "w", encoding="utf-8") as f:
        json.dump(paper_para_embeddings_list, f, indent=4)

    print(f"Total time (sec): {time.time() - start_time}")
