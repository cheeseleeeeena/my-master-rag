from datasets import load_dataset
from pathlib import Path
import json
from transformers import AutoTokenizer
from typing import List, Dict, Union
import time
import re
from vllm import LLM, SamplingParams
import os
import csv


import utils
import evaluator

os.environ["HF_HOME"] = "/workspace/P76125041/.cache/"

READER = "llama3"
READER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HYPOTHESIS = "h1"


# single fact
# low ans-F1 due to noise from retrieved topk (recall=1, precision=1/3, where 1 is gold, 2 is probably noise)
# total cases: 151


# Precomputed prompt template parts
PROMPT_BASE = (
    "Task: Given a question in NLP research field and some relevant snippets, provide an accurate and contextually relevant answer. "
    "The answer can take one of the following forms:\n"
    "- Boolean: 'Yes' or 'No.'\n"
    "- Abstractive: A concise and synthesized answer after reasoning on both the question and the provided content.\n"
    "- Extractive: A direct excerpt from the text.\n"
    "- Unanswerable: If the question cannot be answered based on the provided content.\n\n"
    "Input Details:\n"
    "1. Paper Title: {}\n"
    "2. Relevant Snippets:\n{}\n"
    "3. Question: {}\n\n"
    "Provide a direct and concise answer to the question based only on the relevant snippets. "
    "If the answer is unanswerable, state 'Unanswerable' explicitly.\n\n"
    "Otherwise, the answer to the given question must be enclosed with the special tokens '<answer>' and '</answer>'."
    "Your answer should be as concise as possible.\nAnswer:\n"
)
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an expert in NLP research field.",
}

if __name__ == "__main__":

    # read CSV file to get subset questions
    with open(
        f"eval_results/llama3-{HYPOTHESIS}/{HYPOTHESIS}.csv",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header
        data = list(reader)  # Read remaining rows

    ## Transpose rows to columns, and convert tuples to lists, and unpack into lists
    columns = list(zip(*data))
    columns = [list(col) for col in columns]
    question_ids, annotator_ids = columns
    annotator_ids = list(map(int, annotator_ids))

    # Validate inputs early
    if len(question_ids) != len(annotator_ids):
        raise ValueError("Length of question_subset and annotator_ids must match")

    # Load data once
    raw_test = load_dataset(
        "allenai/qasper",
        split="test",
        cache_dir="/workspace/P76125041/.cache/huggingface/",
    )
    test_questions = utils.load_json(Path("qasper/test_questions.json"))
    test_paper_titles = {paper["id"]: paper["title"] for paper in raw_test}
    gold_data = json.load(open("qasper/test_gold.json"))

    # Filter questions and prepare data efficiently
    filtered_questions = {}
    valid_pairs = []
    for qid, annotator_idx in zip(question_ids, annotator_ids):
        if qid in test_questions:
            filtered_questions[qid] = test_questions[qid]
            valid_pairs.append((qid, annotator_idx))
        else:
            print(
                f"Warning: Question ID {qid} not found in test_questions, skipping..."
            )

    if not valid_pairs:
        raise ValueError("No valid questions found in the subset")

    # Initialize model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL, cache_dir="/workspace/P76125041/.cache"
    )
    model = LLM(model=READER_MODEL, download_dir="/workspace/P76125041/models")
    sampling_params = SamplingParams(max_tokens=500)

    # Compute gold answers and evidence once
    gold_answers_and_evidence = evaluator.get_answers_and_evidence(gold_data, True)

    # Prepare topk_paras efficiently
    all_topk_paras = {}
    for qid, annotator_idx in valid_pairs:
        question_data = filtered_questions[qid]
        print(f"Processing `{question_data['question']}`...")
        evidence = (
            gold_answers_and_evidence[qid][annotator_idx]["evidence"]
            if qid in gold_answers_and_evidence
            and len(gold_answers_and_evidence[qid]) > annotator_idx
            else []
        )
        if not evidence and qid in gold_answers_and_evidence:
            print(
                f"Warning: Annotator index {annotator_idx} invalid for question {qid}, using empty evidence"
            )
        all_topk_paras[qid] = evidence

    # Generate prompts in a single pass
    question_prompts = []
    for qid in filtered_questions:
        question_data = filtered_questions[qid]
        paper_title = test_paper_titles[question_data["from_paper"]]
        concatenated_topk = "\n".join(all_topk_paras[qid])
        user_prompt = PROMPT_BASE.format(
            paper_title, concatenated_topk, question_data["question"]
        )
        messages = [SYSTEM_MESSAGE, {"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        question_prompts.append(formatted_prompt)

    # Timer begins
    start_time = time.time()

    # Batch inference
    outputs = model.generate(question_prompts, sampling_params)

    # Process outputs efficiently
    error_count = 0
    all_predictions = []
    answer_pattern = re.compile(r"\<answer\>(.*?)\<\/answer\>", re.DOTALL)
    for idx, (qid, output) in enumerate(zip(filtered_questions.keys(), outputs)):
        llm_output = output.outputs[0].text
        match = answer_pattern.search(llm_output)
        if match:
            llm_answer = match.group(1).strip()
        else:
            error_count += 1
            llm_answer = llm_output.replace("<answer>", "").replace("</answer>", "")
            print(
                f"No match found for question [{qid}]. Use original answer: {llm_answer}"
            )
        all_predictions.append(
            {
                "question_id": qid,
                "predicted_evidence": all_topk_paras[qid],
                "predicted_answer": llm_answer,
            }
        )

    print(f"total format error count: {error_count}")

    # Save predictions efficiently
    results_dir = Path(f"predictions/llama3-{HYPOTHESIS}")
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = results_dir / "test_predictions.jsonl"
    with open(prediction_file, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(json.dumps(pred, ensure_ascii=False) for pred in all_predictions)
        )

    print(f"total time (sec): {time.time() - start_time}")
