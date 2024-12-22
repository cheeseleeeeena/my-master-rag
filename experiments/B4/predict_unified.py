import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer, util
import jsonlines
import torch
import re
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os
import argparse

import utils

os.environ['HF_HOME'] = '/workspace/P76125041/.cache/'


RETRIEVER = "sbert"
READER = "unified"
MODE = "full"
TOPK = 3

retriever_map: Dict[str, str] = {
    "sbert": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "stella": "dunzhang/stella_en_400M_v5"
}

reader_map: Dict[str, str] = {
    "unified": "allenai/unifiedqa-v2-t5-3b-1363200",
    "unifiedlarge": "allenai/unifiedqa-v2-t5-large-1363200"
}



class UnifiedQAModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     model_name, cache_dir="/workspace/P76125041/.cache/huggingface/"
        # ).to(self.device)
        # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir="/workspace/P76125041/.cache/huggingface/"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]

def format_paragraphs(raw_paras: List[dict], mode: str) -> List[str]:
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
        return [para.get('text', '') for para in raw_paras]


if __name__ == "__main__":
    
    start_time = time.time()

    # load JSON files as dict : paper contents, paper embeddings, questions
    test_questions: Dict[str, Dict] = utils.load_json(Path("qasper/test_questions.json"))
    test_papers: Dict[str, Dict] = utils.load_json(Path("qasper/test_papers.json"))
    paper_para_embeddings: Dict[str, List[List[float]]] = utils.load_json(Path(f"qasper/embeddings/test_embeddings_{RETRIEVER}_{MODE}.json"))
    

    # initialize retriever
    ### Stella
    if RETRIEVER == "stella":
        query_prompt_name: str = "s2p_query"
        embedding_model = SentenceTransformer(
            retriever_map.get(RETRIEVER),
            device="cuda:0",
            cache_folder="/workspace/P76125041/.cache/",
            trust_remote_code=True
        ).cuda(0)
    ### SBERT
    else:
        embedding_model = SentenceTransformer(retriever_map.get(RETRIEVER), cache_folder="/workspace/P76125041/.cache/")
    
    # initialize reader
    ### UNIFIEDQA
    qa_model = UnifiedQAModel(reader_map.get(READER))
    
    
    # prepare question embeddings
    all_questions: List[str] = [q["question"] for q in test_questions.values()]
    ### stella
    if RETRIEVER == "stella":
        question_embeddings = embedding_model.encode(all_questions, prompt_name=query_prompt_name)
    ### sbert
    else:
        question_embeddings = embedding_model.encode(all_questions)
    assert len(question_embeddings)==len(all_questions), "something wrong in question embeddings..."
    
    # prepare output data
    all_predictions: List[Dict] = []
    
    # answer questions    
    for idx, (question_id, question_data) in enumerate(test_questions.items()):
        predictions: Dict[str, Union[str, List[str]]] = {}
        current_question: str = question_data['question']
        print(f"Processing `{current_question}`...")
        
        # ===========================================  RETRIEVER =============================================
        paper_id = question_data["from_paper"]
        
        # get original paragraphs
        raw_paras: List[str] = [para["text"] for para in test_papers[paper_id].values()]
        
        # get embeddings
        q_embedding: List[float] = question_embeddings[idx]
        p_embedding: List[List[float]] = paper_para_embeddings[paper_id]
        
        # compute similarity
        ### stella
        if RETRIEVER == "stella":
            scores: List[float] = embedding_model.similarity(q_embedding, p_embedding)[0]
        ### sbert
        else:
            scores: List[float] = util.dot_score(q_embedding, p_embedding)[0].cpu().tolist()
        assert len(scores)==len(raw_paras), "raw paragraphs & paragraph embeddings mismatched!!!"
        para_score_pairs = list(zip(raw_paras, scores))
        topk_para_score_pairs: List[Tuple[str, float]] = sorted(para_score_pairs, key=lambda x: x[1], reverse=True)[:TOPK]
            # for para, score in topk_para_score_pairs:
            #     print(f"sim score: {score:.3f}")
            #     print(para)
            #     print("=" * 100)
        topk_paras: List[str] = [para for para, _ in topk_para_score_pairs]
        
        predictions["question_id"] = question_id
        predictions["predicted_evidence"] = topk_paras
        
        # ===========================================  READER ===================================================
        
        context: str = "".join(topk_paras)
        answer = qa_model.answer_question(context, current_question)
        predictions["predicted_answer"] = answer
        all_predictions.append(predictions)

    #     # prediction_file: Path = Path("qasper/test_predictions_final.jsonl")
    #     # with open(prediction_file, "a+", encoding="utf-8") as f:
    #     #     f.write(json.dumps(predictions, ensure_ascii=False))
    #     #     f.write("\n")


    results_dir: Path = Path(f"results/{RETRIEVER}-{READER}-{MODE}-top{TOPK}")
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_file: Path = results_dir / "test_predictions.jsonl"
    with open(prediction_file, "w+", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False))
            f.write("\n")
    print(f"total time (sec): {time.time() - start_time}")