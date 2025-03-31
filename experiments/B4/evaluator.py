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

        # process subset questions only
        # if len(predicted) < 1451:
        #     annotator_ids = ["5",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "2",
        #                         "0",
        #                         "2",
        #                         "1",
        #                         "2",
        #                         "1",
        #                         "1",
        #                         "2",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "2",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "2",
        #                         "1",
        #                         "2",
        #                         "2",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "0",
        #                         "1",
        #                         "1",
        #                         "0",
        #                         "1",
        #                         "1",
        #                     ]
        #     annotator_subset = list(map(int, annotator_ids))
        #     qid_subset = [
        #         "cc8b4ed3985f9bfbe1b5d7761b31d9bd6a965444",
        #         "f64449a21c452bc5395a0f0a49fb49825e6385f4",
        #         "91336f12ab94a844b66b607f8621eb8bbd209f32",
        #         "52faf319e37aa15fff1ab47f634a5a584dc42e75",
        #         "6e37f43f4f54ffc77c785d60c6058fbad2147922",
        #         "fff1ed2435ba622d884ecde377ff2de127167638",
        #         "60e6296ca2a697892bd67558a21a83ef01a38177",
        #         "2e4688205c8e344cded7a053b6014cce04ef1bd5",
        #         "2abcff4fdedf9b17f76875cc338ba4ab8d1eccd3",
        #         "38b29b0dcb87868680f9934af71ef245ebb122e4",
        #         "2694a679a703ccd6139897e4d9ff8e053dabd0f2",
        #         "ba39317e918b4386765f88e8c8ae99f9a098c935",
        #         "a79a23573d74ec62cbed5d5457a51419a66f6296",
        #         "4379a3ece3fdb93b71db43f62833f5f724c49842",
        #         "0d9241e904bd2bbf5b9a6ed7b5fc929651d3e28e",
        #         "ae1c4f9e33d0cd64d9a313c318ad635620303cdd",
        #         "8a7a9d205014c42cb0e24a0f3f38de2176fe74c0",
        #         "3e6b6820e7843209495b4f9a72177573afaa4bc3",
        #         "4a32adb0d54da90434d5bd1c66cc03a7956d12a0",
        #         "1d263356692ed8cdee2a13f103a82d98f43d66eb",
        #         "eaacee4246f003d29a108fe857b5dd317287ecf1",
        #         "94d794df4a3109522c2ea09dad5d40e55d35df51",
        #         "1a0794ebbc9ee61bbb7ef2422d576a10576d9d96",
        #         "2fec84a62b4028bbe6500754d9c058eefbc24d9a",
        #         "7c561db6847fb0416bca8a6cb5eebf689a4b1438",
        #         "63d2e97657419a0185127534f4ff9d0039cb1a63",
        #         "475e698a801be0ad9e4f74756d1fff4fe0728009",
        #         "1ec0be667a6594eb2e07c50258b120e693e040a8",
        #         "d6b0c71721ed24ef1d9bd31ed3a266b0c7fc9b57",
        #         "37ac705166fa87dc74fe86575bf04bea56cc4930",
        #         "e6204daf4efeb752fdbd5c26e179efcb8ddd2807",
        #         "47ffc9811b037613c9c4d1ec1e4f13c08396ed1c",
        #         "86d1c990c1639490c239c3dbf5492ecc44ab6652",
        #         "bcf222ad4bb537b01019ed354ea03cd6bf2c1f8e",
        #         "f7f2968feb28c2907266c892f051ae9f7d6286e6",
        #         "477da8d997ff87400c6aad19dcc74f8998bc89c3",
        #         "df4895c6ae426006e75c511a304d56d37c42a1c7",
        #         "99554d0c76fbaef90bce972700fa4c315f961c31",
        #         "11c4071d9d7efeede84f47892b1fa0c6a93667eb",
        #         "e1f61500eb733f2b95692b6a9a53f8aaa6f1e1f6",
        #         "c3befe7006ca81ce64397df654c31c11482dafbe",
        #         "355cf303ba61f84b580e2016fcb24e438abeafa7",
        #         "88757bc49ccab76e587fba7521f0981d6a1af2f7",
        #         "74ebfba06f37cc95dfe59c3790ebe6165e6be19c",
        #         "76e17e648a4d1f386eb6bf61b0c24f134af872be",
        #         "ee19fd54997f2eec7c87c7d4a2169026fe208285",
        #         "13fb28e8b7f34fe600b29fb842deef75608c1478",
        #         "eaae11ffd4ff955de2cd6389b888f5fd2c660a32",
        #         "c0af44ebd7cd81270d9b5b54d4a40feed162fa54",
        #         "e3c44964eb6ddc554901244eb6595f26a9bae47e",
        #         "d207f78beb6cd754268881bf575c8f98000667ea",
        #         "c845110efee2f633d47f5682573bc6091e8f5023",
        #         "1a419468d255d40ae82ed7777618072a48f0091b",
        #         "a01af34c7f630ba0e79e0a0120d2e1c92d022df5",
        #         "21f6cb3819c85312364dd17dd4091df946591ef0",
        #         "452e978bd597411b65be757bf47dc6a78f3c67c9",
        #         "29794bda61665a1fbe736111e107fd181eacba1b",
        #         "4ed58d828cd6bb9beca1471a9fa9f5e77488b1d1",
        #         "1c8958ec50976a9b1088c51e8f73a767fb3973fa",
        #         "c66e0aa86b59bbf9e6a1dc725fb9785473bfa137",
        #         "f8bba20d1781ce2b14fad28d6eff024e5a6c2c02",
        #         "e12166fa9d6f63c4e92252c95c6a7bc96977ebf4",
        #         "5b95665d44666a1dc9e568d2471e5edf8614859f",
        #         "740cc392c0c8bfadfe6b3a60c0be635c03e17f2a",
        #         "00df1ff914956d4d23299d02fd44e4c985bb61fa",
        #         "894bbb1e42540894deb31c04cba0e6cfb10ea912",
        #         "573b8b1ad919d3fd0ef7df84e55e5bfd165b3e84",
        #         "0ba8f04c3fd64ee543b9b4c022310310bc5d3c23",
        #         "5e4eac0b0a73d465d74568c21819acaec557b700",
        #         "70abb108c3170e81f8725ddc1a3f2357be5a4959",
        #         "389cc454ac97609e9d0f2b2fe70bf43218dd8ba7",
        #         "014830892d93e3c01cb659ad31c90de4518d48f3",
        #         "b065a3f598560fdeba447f0a100dd6c963586268",
        #         "a59e86a15405c8a11890db072b99fda3173e5ab2",
        #         "17fd6deb9e10707f9d1b70165dedb045e1889aac",
        #         "8c288120139615532838f21094bba62a77f92617",
        #         "7e9aec2bdf4256c6249cad9887c168d395b35270",
        #         "3138f916e253abed643d3399aa8a4555b2bd8c0f",
        #         "74a17eb3bf1d4f36e2db1459a342c529b9785f6e",
        #         "6656a9472499331f4eda45182ea697a4d63e943c",
        #         "de313b5061fc22e8ffef1706445728de298eae31",
        #         "1397b1c51f722a4ee2b6c64dc9fc6afc8bd3e880",
        #         "cc8bcea4052bf92f249dda276acc5fd16cac6fb4",
        #         "f44a9ed166a655df1d54683c91935ab5e566a04f",
        #         "d14118b18ee94dafe170439291e20cb19ab7a43c",
        #         "d3cfbe497a30b750a8de3ea7f2cecf4753a4e1f9",
        #         "73d87f6ead32653a518fbe8cdebd81b4a3ffcac0",
        #         "4d824b49728649432371ecb08f66ba44e50569e0",
        #         "90756bdcd812b7ecc1c5df2298aa7561fd2eb02c",
        #         "143409d16125790c8db9ed38590a0796e0b2b2e2",
        #         "8ba582939823faae6822a27448ea011ab6b90ed7",
        #         "4c7ec282697f4f6646eb1c19f46bbaf8670b0de6",
        #         "ec5e84a1d1b12f7185183d165cbb5eae66d9833e",
        #         "fe6181ab0aecf5bc8c3def843f82e530347d918b",
        #         "0b1b8e1b583242e5be9b7be73160630a0d4a96b2",
        #         "830f9f9499b06fb4ac3ce2f2cf035127b4f0ec63",
        #         "ee8a77cddbe492c686f5af3923ad09d401a741b5",
        #         "867b1bb1e6a38de525be7757d49928a132d0dbd8",
        #         "c2037887945abbdf959389dc839a86bc82594505",
        #         "1fd31fdfff93d65f36e93f6919f6976f5f172197",
        #         "6657ece018b1455035421b822ea2d7961557c645",
        #         "175cddfd0bcd77b7327b62f99e57d8ea93f8d8ba",
        #         "73a5783cad4ed468a8dbb31b5de2c618ce351ad1",
        #         "a3ba21341f0cb79d068d24de33b23c36fa646752",
        #         "5181527e6a61a9a192db5f8064e56ec263c42661",
        #         "7ef7a5867060f91eac8ad857c186e51b767c734b",
        #         "caebea05935cae1f5d88749a2fc748e62976eab7",
        #         "0a736e0e3305a50d771dfc059c7d94b8bd27032e",
        #         "9f8c0e02a7a8e9ee69f4c1757817cde85c7944bd",
        #         "bba677d1a1fe38a41f61274648b386bdb44f1851",
        #         "b6c2a391c4a94eaa768150f151040bb67872c0bf",
        #         "9ec0527bda2c302f4e82949cc0ae7f7769b7bfb8",
        #         "fb06ed5cf9f04ff2039298af33384ca71ddbb461",
        #         "754d7475b8bf50499ed77328b4b0eeedf9cb2623",
        #         "718c0232b1f15ddb73d40c3afbd6c5c0d0354566",
        #         "361f330d3232681f1a13c6d59abb6c18246e7b35",
        #         "b424ad7f9214076b963a0077d7345d7bb5a7a205",
        #         "c59d67930edd3d369bd51a619849facdd0770644",
        #         "8db11d9166474a0e98b99ac7f81d1f14539d79ec",
        #         "26a321e242e58ea5f2ceaf37f26566dd0d0a0da1",
        #         "f741d32b92630328df30f674af16fbbefcad3f93",
        #         "05118578b46e9d93052e8a760019ca735d6513ab",
        #         "df623717255ea2c9e0f846859d8a9ef51dc1102b",
        #         "ac482ab8a5c113db7c1e5f106a5070db66e7ba37",
        #         "ff36168caf48161db7039e3bd4732cef31d4de99",
        #         "2f23bd86a9e27dcd88007c9058ddfce78a1a377b",
        #         "fe1a74449847755cd7a46647cc9d384abfee789e",
        #         "d622564b250cffbb9ebbe6636326b15ec3c622d9",
        #         "80d6b9123a10358f57f259b8996a792cac08cb88",
        #         "f010f9aa4ba1b4360a78c00aa0747d7730a61805",
        #         "dc473819b196c0ea922773e173a6b283fa778791",
        #         "76ae794ced3b5ae565f361451813f2f3bc85b214",
        #         "d8ae36ae1b4d3af5b59ebd24efe94796101c1c12",
        #         "00f507053c47e55d7e72bebdbd8a75b3ca88cf85",
        #         "86c867b393db0ec4ad09abb48cc1353cac47ea4c",
        #         "6371c6863fe9a14bf67560e754ce531d70de10ab",
        #         "28a8a1542b45f67674a2f1d54fff7a1e45bfad66",
        #         "9b3371dcd855f1d3342edb212efa39dfc9142ae3",
        #         "311f9971d61b91c7d76bba1ad6f038390977a8be",
        #         "e86fb784011de5fda6ff8ccbe4ee4deadd7ee7d6",
        #         "d206f2cbcc3d2a6bd0ccaa3b57fece396159f609",
        #         "ba406e07c33a9161e29c75d292c82a15503beae5",
        #         "b21f61c0f95fefdb1bdb90d51cbba4655cd59896",
        #         "8476d0bf5962f4ed619a7b87415ebe28c38ce296",
        #         "cf68906b7d96ca0c13952a6597d1f23e5184c304",
        #         "e6f5444b7c08d79d4349e35d5298a63bb30e7004",
        #         "59f41306383dd6e201bded0f1c7c959ec4f61c5a",
        #         "cfcf94b81589e7da215b4f743a3f8de92a6dda7a",
        #         "81e101b2c803257492d67a00e8a1d9a07cbab136",
        #         "b942d94e4187e4fdc706cfdf92e3a869fc294911",
        #         "55e3daecaf8030ed627e037992402dd0a7dd67ff",
        #     ]

        #     for subset_qid, annotator in zip (qid_subset, annotator_subset):

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
            "evidence_precision": evidence_precision,  # Added
            "evidence_recall": evidence_recall,  # Added
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
    analysis_file = results_dir / "detailed_analysis_0331.csv"
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
