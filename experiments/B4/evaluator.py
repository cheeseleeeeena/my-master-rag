from collections import Counter
import argparse
import string
import re
import json
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

TOPK = 3


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# return precision and recall scores separately
def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        return 1.0, 1.0, 1.0  # f1, precision, recall
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0, 0.0, 0.0  # f1, precision, recall
    precision = num_same / len(prediction) if prediction else 0.0
    recall = num_same / len(ground_truth) if ground_truth else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall


def categorize_answer(answer: str, evidences: list[str]) -> str:
    answer: str = normalize_answer(answer)

    if "unanswerable" in answer:
        if answer == "unanswerable":
            return "none"
    elif answer == "yes" or answer == "no":
        return "boolean"
    else:
        evidence_txt: str = "".join(evidences)
        if answer in evidence_txt.strip():
            return "extractive"
        else:
            return "abstractive"


def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append(
                        {"answer": "Unanswerable", "evidence": [], "type": "none"}
                    )
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(
                            f"Annotation {answer_info['annotation_id']} does not contain an answer"
                        )
                    if text_evidence_only:
                        evidence = [
                            text
                            for text in answer_info["evidence"]
                            if "FLOAT SELECTED" not in text
                        ]
                    else:
                        evidence = answer_info["evidence"]
                    references.append(
                        {"answer": answer, "evidence": evidence, "type": answer_type}
                    )
            answers_and_evidence[question_id] = references
    return answers_and_evidence


# get originating section of given paragraph
def get_sections(
    predicted_paras: List[str], gold_data: Dict, paper_id: str
) -> List[str]:
    full_text: List[Dict] = gold_data[paper_id]["full_text"]
    section_names = [section["section_name"] for section in full_text]
    section_paragraphs = [section["paragraphs"] for section in full_text]
    para_to_section: Dict[str, str] = {}
    for section_name, paragraphs in zip(section_names, section_paragraphs):
        for para in paragraphs:
            para_to_section[para] = section_name
    predicted_sections: List[str] = []
    for predicted_para in predicted_paras:
        predicted_sections.append(para_to_section.get(predicted_para, None))
    return predicted_sections


# added new functionality:
## record the index of the final answer with highest score of each question
## record the precision and recall scores of the chosen max_f1_score
### (to further analyze how the f1 scores is affected by prec & rec )
def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_evidence_precisions = []
    max_evidence_recalls = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    max_score_details = {}
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            max_evidence_precisions.append(0.0)
            max_evidence_recalls.append(0.0)
            max_score_details[question_id] = {
                "answer_index": None,
                "evidence_index": None,
            }
            continue

        # calculate Answer-F1 scores for all reference answers
        predicted_answer_tuples = [
            (
                token_f1_score(predicted[question_id]["answer"], reference["answer"]),
                reference["type"],
                idx,
            )
            for idx, reference in enumerate(gold[question_id])
        ]

        max_answer_tuple = sorted(
            predicted_answer_tuples, key=lambda x: x[0], reverse=True
        )[0]
        max_answer_f1, answer_type, max_answer_idx = max_answer_tuple
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)

        # calculate Evidence-F1 scores for all reference answers
        predicted_evidence_tuples = [
            (
                paragraph_f1_score(
                    predicted[question_id]["evidence"], reference["evidence"]
                ),
                idx,
            )
            for idx, reference in enumerate(gold[question_id])
        ]
        # Extract max evidence F1, precision, recall based on F1 score
        max_evidence_tuple = sorted(
            predicted_evidence_tuples,
            key=lambda x: x[0][0],
            reverse=True,  # Sort by F1
        )[0]
        (
            max_evidence_f1,
            max_evidence_precision,
            max_evidence_recall,
        ), max_evidence_idx = max_evidence_tuple

        max_evidence_f1s.append(max_evidence_f1)
        max_evidence_precisions.append(max_evidence_precision)
        max_evidence_recalls.append(max_evidence_recall)

        max_score_details[question_id] = {
            "answer_f1": max_answer_f1,
            "answer_index": max_answer_idx,
            "evidence_f1": max_evidence_f1,
            "evidence_precision": max_evidence_precision,
            "evidence_recall": max_evidence_recall,
            "evidence_index": max_evidence_idx,
        }
    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {
            key: mean(value) for key, value in max_answer_f1s_by_type.items()
        },
        "Evidence F1": mean(max_evidence_f1s),
        "Evidence Precision": mean(max_evidence_precisions),  # Added
        "Evidence Recall": mean(max_evidence_recalls),  # Added
        "Missing predictions": num_missing_predictions,
        "Max Score Details": max_score_details,
    }


def get_detailed_analysis(
    gold_data: Dict, predicted: Dict, text_evidence_only: bool
) -> List[Dict]:
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, text_evidence_only)
    eval_results = evaluate(gold_answers_and_evidence, predicted)
    max_score_details = eval_results["Max Score Details"]
    qid_to_details = {}
    for paper_id, paper_info in gold_data.items():
        for qa_info in paper_info["qas"]:
            qid_to_details[qa_info["question_id"]] = {
                "paper_id": paper_id,
                "question": qa_info["question"],
                "paper_title": paper_info["title"],
            }
    detailed_analysis = []
    for question_id, references in gold_answers_and_evidence.items():
        best_answer = max_score_details.get(
            question_id,
            {
                "answer_f1": None,
                "answer_index": None,
                "evidence_f1": None,
                "evidence_precision": None,  # Added
                "evidence_recall": None,  # Added
                "evidence_index": None,
            },
        )
        answer_idx = best_answer["answer_index"]
        evidence_idx = best_answer["evidence_index"]
        answer_f1 = best_answer["answer_f1"]
        evidence_f1 = best_answer["evidence_f1"]
        evidence_precision = best_answer["evidence_precision"]  # Added
        evidence_recall = best_answer["evidence_recall"]  # Added
        gold_answer = (
            references[answer_idx]["answer"] if answer_idx is not None else None
        )
        gold_answer_type = (
            references[answer_idx]["type"] if answer_idx is not None else None
        )
        gold_paras = (
            references[evidence_idx]["evidence"] if evidence_idx is not None else []
        )
        pred_info = predicted.get(question_id, {"answer": None, "evidence": []})
        predicted_answer = pred_info["answer"]
        predicted_paras = pred_info["evidence"]
        paper_details = qid_to_details.get(
            question_id, {"paper_id": None, "question": None, "paper_title": None}
        )
        paper_id = paper_details["paper_id"]
        # gold_sections = (
        #     get_sections(gold_paras, gold_data, paper_id)
        #     if paper_id and gold_paras
        #     else []
        # )
        # predicted_sections = (
        #     get_sections(predicted_paras, gold_data, paper_id)
        #     if paper_id and predicted_paras
        #     else []
        # )

        # categorize predicted answer type
        predicted_ans_type: str = categorize_answer(predicted_answer, predicted_paras)
        analysis_entry = {
            "qid": question_id,
            "question": paper_details["question"],
            "from_paper": paper_id,
            "paper_title": paper_details["paper_title"],
            "token_f1": answer_f1,
            "evidence_f1": evidence_f1,
            "evidence_precision": evidence_precision,  # Added
            "evidence_recall": evidence_recall,  # Added
            "best_ev_idx": evidence_idx,
            "best_ans_idx": answer_idx,
            # get the first 3 paras only since the maximum is 52 is too much
            "gold_paras": gold_paras[:TOPK],
            "full_gold_paras": gold_paras,
            # "gold_sections": gold_sections[:3],
            # "full_gold_sections": gold_sections,
            "predicted_paras": predicted_paras,
            # "predicted_sections": predicted_sections,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "gold_answer_type": gold_answer_type,
            "pred_answer_type": predicted_ans_type,
        }
        detailed_analysis.append(analysis_entry)
    return detailed_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset",
    )
    parser.add_argument(
        "--text_evidence_only", action="store_true", help="Ignore non-text evidence"
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Subdirectory name under 'predictions' to read prediction file and under 'eval_results' to save outputs",
    )
    args = parser.parse_args()

    # Load gold data
    gold_data = json.load(open(args.gold))

    # Load predictions
    predicted_answers_and_evidence = {}
    prediction_file: Path = (
        "predictions" / Path(args.settings) / "test_predictions.jsonl"
    )
    for line in open(prediction_file):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": prediction_data["predicted_answer"],
            "evidence": prediction_data["predicted_evidence"],
        }

    # Get evaluation results
    gold_answers_and_evidence = get_answers_and_evidence(
        gold_data, args.text_evidence_only
    )
    evaluation_output = evaluate(
        gold_answers_and_evidence, predicted_answers_and_evidence
    )

    # Get detailed analysis
    detailed_analysis = get_detailed_analysis(
        gold_data, predicted_answers_and_evidence, args.text_evidence_only
    )

    # Create results directory and subdirectory
    results_dir = Path("eval_results") / args.settings
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation results as JSON
    eval_file = results_dir / "evaluation_scores.json"
    with open(eval_file, "w") as f:
        json.dump(evaluation_output, f, indent=2)
    print(f"Evaluation Results saved to {eval_file}")

    # Determine max number of paragraphs for gold and predicted
    max_gold_paras = (
        max(len(entry["gold_paras"]) for entry in detailed_analysis)
        if detailed_analysis
        else 0
    )
    max_pred_paras = (
        max(len(entry["predicted_paras"]) for entry in detailed_analysis)
        if detailed_analysis
        else 0
    )

    # Prepare data for CSV with separate columns for each paragraph and section
    csv_data = []

    for entry in detailed_analysis:
        row = {
            "qid": entry["qid"],
            "question": entry["question"],
            "paper_id": entry["from_paper"],
            "paper_title": entry["paper_title"],
            "ans_f1": entry["token_f1"],
            "ev_f1": entry["evidence_f1"],
            "ev_precision": entry["evidence_precision"],  # Added
            "ev_recall": entry["evidence_recall"],  # Added
            "ans_type": entry["gold_answer_type"],
            "pred_ans_type": entry["pred_answer_type"],
            "best_ans_id": entry["best_ans_idx"],
            "best_ev_id": entry["best_ev_idx"],
            "gold_answer": entry["gold_answer"],
            "predicted_answer": entry["predicted_answer"],
        }
        # Add gold paragraphs and sections as separate columns
        for i in range(max_gold_paras):
            row[f"gold_para_{i+1}"] = (
                entry["gold_paras"][i] if i < len(entry["gold_paras"]) else ""
            )
        # Add predicted paragraphs and sections as separate columns
        for i in range(max_pred_paras):
            row[f"predicted_para_{i+1}"] = (
                entry["predicted_paras"][i] if i < len(entry["predicted_paras"]) else ""
            )
        csv_data.append(row)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_data)
    analysis_file = results_dir / "detailed_analysis_0414.csv"
    df.to_csv(analysis_file, index=False)
    print(f"Detailed Analysis saved to {analysis_file}")

    # Optional: Print to console
    # print("\nEvaluation Results:")
    # print(json.dumps(evaluation_output, indent=2))
    # print("\nDetailed Analysis (first few rows):")
    # print(df.head().to_string())

    # print(f" [{max_qid}] max_gold_paras_count = {max_gold_paras_count}")
    gold_paras_counts: List[Tuple[str, str, int]] = [
        (entry["qid"], entry["gold_answer_type"], len(entry["full_gold_paras"]))
        for entry in detailed_analysis
    ]
    truncated_gold_paras_counts: List[Tuple[str, int]] = []
    multi_gold_paras_counts: List[Tuple[str, int]] = []
    single_gold_paras: List[str] = []
    unanswerable_counts: List[str] = []
    unanswerable_gold_paras_counts: List[Tuple[str, int]] = []
    outliers: List[str] = []
    answerable_without_gold_paras: List[str] = []

    for qid, ans_type, count in gold_paras_counts:
        if ans_type == "none":
            if count == 0:
                # unanswerable questions and did not provide gold evidences
                unanswerable_counts.append(qid)
            else:
                # unanswerable questions with gold evidences
                unanswerable_gold_paras_counts.append((qid, count))
        elif count == 0:
            # exception: gold evidences not provided
            answerable_without_gold_paras.append(qid)
        # multi-fact (unaligned)
        elif count > 1:
            if count > 3:
                truncated_gold_paras_counts.append((qid, count))
            else:
                multi_gold_paras_counts.append(count)
        elif count == 1:
            single_gold_paras.append(qid)
        else:
            outliers.append(qid)

    print(f"total # question: {len(gold_paras_counts)}")
    print(f"# unanswerable question = {len(unanswerable_counts)}")
    print(
        f"# unanswerable question with gold paras = {len(unanswerable_gold_paras_counts)}"
    )

    print(
        f"# answerable question but no annotated gold paras = {len(answerable_without_gold_paras)}"
    )
    print(f"# truncated gold paras = {len(truncated_gold_paras_counts)}")
    print(f"# question with multi-evidences: {len(multi_gold_paras_counts)}")
    print(f"# question with single evidence: {len(single_gold_paras)}")
    assert len(multi_gold_paras_counts) + len(single_gold_paras) + len(outliers) + len(
        unanswerable_gold_paras_counts
    ) + +len(unanswerable_counts) + len(answerable_without_gold_paras) + len(
        truncated_gold_paras_counts
    ) == len(
        gold_paras_counts
    ), "something went wrong while categorizing!"
    max_truncated_count: int = np.max([obj[1] for obj in truncated_gold_paras_counts])
    min_truncated_count: int = np.min([obj[1] for obj in truncated_gold_paras_counts])
    print(f"max truncated evidence count: {max_truncated_count}")
    print(f"min truncated evidence count: {min_truncated_count}")
    print(
        f"median of evidence count among multihop question: {np.median(multi_gold_paras_counts)}"
    )
    print(
        f"mode of evidence count among multihop question: {stats.mode(multi_gold_paras_counts)}"
    )
