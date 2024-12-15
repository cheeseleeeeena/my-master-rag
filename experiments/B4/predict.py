import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import jsonlines
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class UnifiedQAModel:
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir="/workspace/P76125041/.cache/huggingface/"
        ).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

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


if __name__ == "__main__":

    data_file: Path = Path("qasper/test_papers.json")
    questions_file: Path = Path("qasper/test_questions.json")
    full_title: bool = True

    # get all paper contents from test set
    with open(data_file, "r") as file:
        test_papers: Dict[str, Dict] = json.load(file)

    # get all questions from test set
    with open(questions_file, "r") as file:
        test_questions: Dict[str, Dict] = json.load(file)

    query_prompt_name = "s2p_query"
    # model = SentenceTransformer(
    #     "dunzhang/stella_en_400M_v5", trust_remote_code=True, device="cuda:1"
    # ).cuda(1)
    model = SentenceTransformer(
        "/workspace/P76125041/models/stella_en_400M_v5",
        trust_remote_code=True,
        device="cuda:1",
    ).cuda(1)
    topk = 10

    qa_model = UnifiedQAModel()
    all_predictions: List[Dict] = []

    # process every question
    for question_id, question_data in test_questions.items():

        # prepare prediction data
        predictions: Dict[str, Union[str, List[str]]] = {}

        # get all paragraphs from the corresponding paper
        paper_data: Dict[str, Dict] = test_papers[question_data["from_paper"]]
        question_text: str = question_data["question"]

        print(f"Processing `{question_text}`...")
        # generate docs (paragraphs) for encode
        if full_title:
            para_ids: List[int] = [int(pid) for pid in paper_data.keys()]
            paragraphs: List[str] = [
                f"Section: {para.get('section_name', 'Unnamed Section')}\n\nText: {para.get('text', '')}"
                for para in paper_data.values()
            ]

        query_embedding = model.encode([question_text], prompt_name=query_prompt_name)
        para_embeddings = model.encode(paragraphs)
        # print(query_embedding.shape, para_embeddings.shape)
        # (2, 1024) (2, 1024)

        similarities = model.similarity(query_embedding, para_embeddings)
        para_scores: List[Tuple[int, str, float]] = list(
            zip(para_ids, paragraphs, similarities[0].tolist())
        )
        topk_para_scores: List[Tuple[int, str, float]] = sorted(
            para_scores, key=lambda x: x[2], reverse=True
        )[:topk]
        topk_paras: List[str] = [para for _, para, _ in topk_para_scores]
        # print(question_id)
        # print("*" * 100)
        predictions["question_id"] = question_id
        predictions["predicted_evidence"] = topk_paras
        context = "\n".join(topk_paras)
        # for id, para, score in topk_paras_with_score:
        #     print(f"para ID: {id}")
        #     print(f"sim score: {score:.3f}")
        #     print(para)
        #     print("=" * 100)

        answer = qa_model.answer_question(context, question_text)
        predictions["predicted_answer"] = answer

        all_predictions.append(predictions)

    prediction_file: Path = Path("qasper/test_predictions.jsonl")
    with open(prediction_file, "w+", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False))
            f.write("\n")
