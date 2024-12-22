from datasets import load_dataset
from pathlib import Path
import time
import json
from typing import List, Dict
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

import os

os.environ["HF_HOME"] = "/workspace/P76125041/.cache/"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/workspace/P76125041/.cache/",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir="/workspace/P76125041/.cache/"
)


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

if __name__ == "__main__":
    raw_test = load_dataset(
        "allenai/qasper",
        split="test",
    )

    # Prepare LLM for summarization
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # model = LLM(
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",
    #     download_dir="/workspace/P76125041/models",
    # )
    # sampling_params = SamplingParams(max_tokens=100)

    questions_file: Path = Path("qasper/test_questions.json")

    full_papers = {}
    # get all paper contents from test set
    for paper in raw_test:
        paper_id = paper["id"]
        paper_title = paper["title"]
        paragraphs = paper["full_text"]["paragraphs"]
        section_names = paper["full_text"]["section_name"]

        full_papers[paper_id] = {}
        full_papers[paper_id]["title"] = paper_title

        paper_texts: List[str] = []
        for section_idx, paras_in_section in enumerate(paragraphs):
            section_title = section_names[section_idx]
            section_text = (
                f"<heading>{section_title}</heading>\n"
                + "\n".join(paras_in_section).strip()
            )
            paper_texts.append(section_text)
        full_papers[paper_id]["text"] = "".join(paper_texts)

    # get all questions from test set
    with open(questions_file, "r") as file:
        test_questions: Dict[str, Dict] = json.load(file)

    # prepare question prompts for llama batch inference
    question_ids: List[str] = list(test_questions.keys())
    question_prompts: List[str] = []
    all_messages: List[List[Dict[str, str]]] = []
    for question_id, question_data in test_questions.items():

        # get all paragraphs from the corresponding paper
        paper_id: Dict[str, Dict] = question_data["from_paper"]
        question_text: str = question_data["question"]

        user_prompt = (
            "Task: Given a research paper and a corresponding question derived from it, provide an accurate and contextually relevant answer. "
            "The answer can take one of the following forms:\n"
            "- Boolean: 'Yes' or 'No.'\n"
            "- Abstractive: A concise and synthesized answer after reasoning on both the question and the provided content.\n"
            "- Extractive: A direct excerpt from the text.\n"
            "- Unanswerable: If the question cannot be answered based on the provided content.\n\n"
            "Input Details:\n"
            f"1. Paper Title: {full_papers[paper_id]['title']}\n"
            f"2. Paper Contents:\n{full_papers[paper_id]['text']}\n"
            f"3. Question: {question_text}\n\n"
            "Provide a direct and concise answer to the question based only on the given paper's contents. "
            "If the answer is unanswerable, state 'Unanswerable' explicitly.\n\n"
            "Otherwise, the answer to the given question must be enclosed with '<<<' and '>>>'."
            "Answer:\n"
        )

        messages = [
            {"role": "system", "content": "You are an expert in NLP research field."},
            {"role": "user", "content": user_prompt},
        ]

        # formatted_prompt = tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # question_prompts.append(formatted_prompt)

        # question_prompts.append(messages)
        all_messages.append(messages)

    prompts = pipeline.tokenizer.apply_chat_template(
        all_messages, tokenize=False, add_generation_prompt=True
    )

    # save model outputs
    # outputs = model.generate(question_prompts, sampling_params)
    start_time = time.time()
    outputs = pipeline(
        prompts,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    all_predictions: List[Dict[str, str]] = []
    for idx, output in enumerate(outputs):
        prediction = {}
        prediction["question_id"] = question_ids[idx]
        prediction["predicted_evidence"] = ["None"]
        llm_output = output["generated_text"][-1]
        match = re.search(r"<<<(.*?)>>>", llm_output, re.DOTALL)

        # Extract the match if found
        if match:
            llm_answer = match.group(1).strip()
            # print(extracted_text)
        else:
            print("No match found.")
            llm_answer = ""
        prediction["predicted_answer"] = llm_answer
        # all_predictions.append({
        #     "question_id" : question_ids[idx]
        #     "predicted_evidence" : ["None"]
        #     "predicted_answer" : output.outputs[0].text
        # })

        prediction_file: Path = Path("qasper/test_predictions_acc.jsonl")
        with open(prediction_file, "a+", encoding="utf-8") as f:
            f.write(json.dumps(prediction, ensure_ascii=False))
            f.write("\n")

        all_predictions.append(prediction)

    final_prediction_file: Path = Path("qasper/test_predictions_llama.jsonl")
    with open(final_prediction_file, "w+", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False))
            f.write("\n")
    print(f"Total time (sec): {time.time() - start_time}")
