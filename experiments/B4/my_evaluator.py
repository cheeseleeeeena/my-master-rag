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


# unchanged
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


# unchanged
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


# unchanged
def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# unchanged
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
def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
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
            max_score_details[question_id] = {
                "answer_index": None,
                "evidence_index": None,
            }
            continue
        answer_f1s_and_types = [
            (
                token_f1_score(predicted[question_id]["answer"], reference["answer"]),
                reference["type"],
                idx,
            )
            for idx, reference in enumerate(gold[question_id])
        ]
        max_answer_tuple = sorted(
            answer_f1s_and_types, key=lambda x: x[0], reverse=True
        )[0]
        max_answer_f1, answer_type, max_answer_idx = max_answer_tuple
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        evidence_f1s_and_indices = [
            (
                paragraph_f1_score(
                    predicted[question_id]["evidence"], reference["evidence"]
                ),
                idx,
            )
            for idx, reference in enumerate(gold[question_id])
        ]
        max_evidence_tuple = sorted(
            evidence_f1s_and_indices, key=lambda x: x[0], reverse=True
        )[0]
        max_evidence_f1, max_evidence_idx = max_evidence_tuple
        max_evidence_f1s.append(max_evidence_f1)
        max_score_details[question_id] = {
            "answer_f1": max_answer_f1,
            "answer_index": max_answer_idx,
            "evidence_f1": max_evidence_f1,
            "evidence_index": max_evidence_idx,
        }
    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {
            key: mean(value) for key, value in max_answer_f1s_by_type.items()
        },
        "Evidence F1": mean(max_evidence_f1s),
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
                "evidence_index": None,
            },
        )
        answer_idx = best_answer["answer_index"]
        evidence_idx = best_answer["evidence_index"]
        answer_f1 = best_answer["answer_f1"]
        evidence_f1 = best_answer["evidence_f1"]
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
        gold_sections = (
            get_sections(gold_paras, gold_data, paper_id)
            if paper_id and gold_paras
            else []
        )
        predicted_sections = (
            get_sections(predicted_paras, gold_data, paper_id)
            if paper_id and predicted_paras
            else []
        )
        analysis_entry = {
            "qid": question_id,
            "question": paper_details["question"],
            "from_paper": paper_id,
            "paper_title": paper_details["paper_title"],
            "token_f1": answer_f1,
            "evidence_f1": evidence_f1,
            "best_ev_idx": evidence_idx,
            "best_ans_idx": answer_idx,
            # get the first 3 paras only since the maximum is 52 is too much
            "gold_paras": gold_paras[:3],
            "full_gold_paras": gold_paras,
            "gold_sections": gold_sections[:3],
            "full_gold_sections": gold_sections,
            "predicted_paras": predicted_paras,
            "predicted_sections": predicted_sections,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "gold_answer_type": gold_answer_type,
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
        help="Subdirectory name under 'results' to save outputs",
    )
    args = parser.parse_args()

    # Load gold data
    gold_data = json.load(open(args.gold))

    # Load predictions
    predicted_answers_and_evidence = {}
    prediction_file: Path = "results" / Path(args.settings) / "test_predictions.jsonl"
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
    results_dir = Path("results") / args.settings
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation results as JSON
    eval_file = results_dir / "evaluation_results.json"
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

    # max_gold_paras_count = 0
    # max_qid = ""
    # for question_id, references in gold_answers_and_evidence.items():
    #     for i, possible_ans in enumerate(references):
    #         gold_paras_count = len(possible_ans["evidence"])
    #         if gold_paras_count > max_gold_paras_count:
    #             max_gold_paras_count = gold_paras_count
    #         print(
    #             f"[{question_id}] # of gold paras for annotator {i}: {gold_paras_count}"
    #         )
    for entry in detailed_analysis:
        # if len(entry["gold_paras"]) > max_gold_paras_count:
        #     max_gold_paras_count = len(entry["gold_paras"])
        #     max_qid = entry["qid"]
        row = {
            "qid": entry["qid"],
            "question": entry["question"],
            "paper_id": entry["from_paper"],
            "paper_title": entry["paper_title"],
            "ans_f1": entry["token_f1"],
            "ev_f1": entry["evidence_f1"],
            "ans_type": entry["gold_answer_type"],
            "best_ans_id": entry["best_ans_idx"],
            "best_ev_id": entry["best_ev_idx"],
            "gold_answer": entry["gold_answer"],
            "predicted_answer": entry["predicted_answer"],
            # "gold_sections": (
            #     " | ".join(str(s) for s in entry["gold_sections"])
            #     if entry["gold_sections"]
            #     else ""
            # # ),
            # "predicted_sections": (
            #     " | ".join(str(s) for s in entry["predicted_sections"])
            #     if entry["predicted_sections"]
            #     else ""
            # ),
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
    analysis_file = results_dir / "detailed_analysis.csv"
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
                unanswerable_counts.append(qid)
            else:
                unanswerable_gold_paras_counts.append((qid, count))
        elif count == 0:
            answerable_without_gold_paras.append(qid)

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
    # print(
    #     f"avg of evidence count among multihop question: {np.average(multi_gold_paras_counts): .2f}"
    # )
    print(
        f"median of evidence count among multihop question: {np.median(multi_gold_paras_counts)}"
    )
    print(
        f"mode of evidence count among multihop question: {stats.mode(multi_gold_paras_counts)}"
    )

    # gather cases for multi-hop outliers
    print("-----")
    print("multi-hop outliers:")
    print("-----")
    multihop_outliers = []
    multihop_outliers_unaligned = []
    for entry in detailed_analysis:
        paras_count = len(entry["full_gold_paras"])
        if (
            entry["gold_answer_type"] == "extractive"
            and paras_count > 3
            and paras_count < 10
        ):
            if entry["best_ev_idx"] == entry["best_ans_idx"]:
                multihop_outliers.append(
                    (entry["qid"], paras_count, entry["best_ans_idx"])
                )
            else:
                multihop_outliers_unaligned.append(
                    (
                        entry["qid"],
                        paras_count,
                    )
                )

    if not multihop_outliers:
        print("all extractive cases have no outlier!")
    else:
        print(f"total outlier: {len(multihop_outliers)}")
        for qid, count, idx in multihop_outliers:
            print(f"[{qid}] count={count}, chosen idx={idx}")
            print("----")
    print(f"total unaligned multihop outliers: {len(multihop_outliers_unaligned)}")

    # gather cases for extractive type
    # print("-----")
    # print("extractive stats:\n")
    # print("-----")
    # extractive_cases: Dict = {}
    # extractive_cases["unannotated"] = []
    # for entry in detailed_analysis:
    #     if entry["gold_answer_type"] == "extractive" and len(entry["gold_paras"]) == 0:
    #         if entry["best_ev_idx"] == entry["best_ans_idx"]:
    #             extractive_cases["unannotated"].append(
    #                 {
    #                     "qid": entry["qid"],
    #                     "best_ans": entry["best_ev_idx"],
    #                 }
    #             )
    #         else:
    #             extractive_cases["unannotated"].append(
    #                 {
    #                     "qid": entry["qid"],
    #                     "best_ans": (entry["best_ev_idx"], entry["best_ans_idx"]),
    #                 }
    #             )
    # if not extractive_cases["unannotated"]:
    #     print("all extractive cases have proper annotator answer!")
    # else:
    #     print(f"total unannotated: {len(extractive_cases['unannotated'])}")
    #     for case in extractive_cases["unannotated"]:
    #         print(case["qid"])
    #         print(case["best_ans"])
    #         print("----")
    # print(f"total unannotated: {len(extractive_cases['unannotated'])}")
