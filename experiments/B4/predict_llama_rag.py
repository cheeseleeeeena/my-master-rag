# (vllm) llama3-8b
    # format error 71
    # process time: 2min


from datasets import load_dataset
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple, Optional, Union
import time
from sentence_transformers import SentenceTransformer, util
import torch
import re
from vllm import LLM, SamplingParams
import os

import utils

os.environ["HF_HOME"] = "/workspace/P76125041/.cache/"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"4

RETRIEVER = "sbert"
READER = "llama3"
MODE = "full"
TOPK = 3

retriever_map: Dict[str, str] = {
    "sbert": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "stella": "dunzhang/stella_en_400M_v5"
}

reader_map: Dict[str, str] = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}


if __name__ == "__main__":
    
    raw_test = load_dataset(
        "allenai/qasper",
        split="test",
        cache_dir="/workspace/P76125041/.cache/huggingface/",
    )
    
    # load JSON files as dict : paper contents, paper embeddings, questions
    test_questions: Dict[str, Dict] = utils.load_json(Path("qasper/test_questions.json"))
    test_papers: Dict[str, Dict] = utils.load_json(Path("qasper/test_papers.json"))
    test_paper_titles: Dict[str, str] = {paper["id"]: paper["title"] for paper in raw_test}
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

    # initialize LLM reader
    tokenizer = AutoTokenizer.from_pretrained(
        reader_map.get(READER),
        cache_dir="/workspace/P76125041/.cache",
    )
    model = LLM(
        model=reader_map.get(READER),
        download_dir="/workspace/P76125041/models",
    )
    sampling_params = SamplingParams(max_tokens=500)
    
    # prepare question embeddings
    all_questions: List[str] = [q["question"] for q in test_questions.values()]
    ### stella
    if RETRIEVER == "stella":
        question_embeddings = embedding_model.encode(all_questions, prompt_name=query_prompt_name)
    ### sbert
    else:
        question_embeddings = embedding_model.encode(all_questions)
    assert len(question_embeddings)==len(all_questions), "something wrong in question embeddings..."
    
    
    # ===========================================  RETRIEVER =============================================
    
    # prepare topk_paras (for generating prompts LLM batch inference)
    all_topk_paras: Dict[str, List[str]] = {}
    for idx, (question_id, question_data) in enumerate(test_questions.items()):
        current_question: str = question_data['question']
        print(f"Processing `{current_question}`...")
        
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
        all_topk_paras[question_id] = topk_paras


    # format prompts for llama batch inference
    question_prompts: List[str] = []
    for question_id, question_data in test_questions.items():

        # get all paragraphs from the corresponding paper
        paper_id: Dict[str, Dict] = question_data["from_paper"]
        paper_title: str = test_paper_titles[paper_id]
        concatenated_topk: str = "\n".join(all_topk_paras[question_id])
        question_text: str = question_data["question"]

        user_prompt = (
            "Task: Given a question in NLP research field and some relevant snippets, provide an accurate and contextually relevant answer. "
            "The answer can take one of the following forms:\n"
            "- Boolean: 'Yes' or 'No.'\n"
            "- Abstractive: A concise and synthesized answer after reasoning on both the question and the provided content.\n"
            "- Extractive: A direct excerpt from the text.\n"
            "- Unanswerable: If the question cannot be answered based on the provided content.\n\n"
            "Input Details:\n"
            f"1. Paper Title: {paper_title}\n"
            f"2. Relevant Snippets:\n{concatenated_topk}\n"
            f"3. Question: {question_text}\n\n"
            "Provide a direct and concise answer to the question based only on the relevant snippets. "
            "If the answer is unanswerable, state 'Unanswerable' explicitly.\n\n"
            "Otherwise, the answer to the given question must be enclosed with the special tokens '<answer>' and '</answer>'."
            "Your answer should be as concise as possible."
            "Answer:\n"
        )

        messages = [
            {"role": "system", "content": "You are an expert in NLP research field."},
            {"role": "user", "content": user_prompt},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        question_prompts.append(formatted_prompt)

    # timer begins
    start_time = time.time()

    # save model outputs
    outputs = model.generate(question_prompts, sampling_params)

    # record error count where llm did not follow output format
    error_count: int = 0
    
    # process answers question by question
    question_ids: List[str] = list(test_questions.keys())
    all_predictions: List[Dict[str, Union[str, List[str]]]] = []
    for idx, output in enumerate(outputs):
        prediction: Dict[str, Union[str, List[str]]] = {}
        question_id: str = question_ids[idx]
        prediction["question_id"] = question_id
        prediction["predicted_evidence"] = all_topk_paras[question_id]
        llm_output = output.outputs[0].text
        match = re.search(r"\<answer\>(.*?)\<\/answer\>", llm_output, re.DOTALL)

        # Extract the match if found
        if match:
            llm_answer = match.group(1).strip()
            # print(extracted_text)
        else:
            error_count += 1
            llm_answer = re.sub(r"\<answer\>", "", llm_output)
            llm_answer = re.sub(r"\<\/answer\>", "", llm_answer)
            print(
                f"No match found for question [{question_ids[idx]}]. Use original answer: {llm_answer}"
            )
        prediction["predicted_answer"] = llm_answer
        all_predictions.append(prediction)

    print(f"total format error count: {error_count}")
    
    # save predictions as JSONL file
    results_dir: Path = Path(f"results/{RETRIEVER}-{READER}-{MODE}-top{TOPK}")
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_file: Path = results_dir / "test_predictions.jsonl"
    with open(prediction_file, "w+", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False))
            f.write("\n")
    print(f"total time (sec): {time.time() - start_time}")
