from datasets import load_dataset
from pathlib import Path
import json
from typing import List, Dict

if __name__ == "__main__":
    raw_test = load_dataset(
        "allenai/qasper",
        split="test",
        cache_dir="/workspace/P76125041/.cache/huggingface/",
    )

    # Create a Path object for the output directory
    output_dir = Path(f"raptor/qasper/")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw dataset format
    # {
    #     "id": "paper_001",
    #     "title": "Sample Paper 1 ",
    #     "abstract": "This is a sample abstract.",
    #     "full_text": {
    #         "paragraphs": [["Introduction paragraph 1", "Introduction paragraph 2"], ["Methods paragraph 1"]],
    #         "section_name": ["Introduction ::: Background ::: Details", "Methods ::: Experiment ::: Steps"]
    #     },
    # },
    # {
    #     "id": "paper_002",
    #     "title": "Sample Paper 2",
    #     "abstract": "This is a sample abstract.",
    #     "full_text": {
    #         "paragraphs": [["Introduction paragraph 1", "Introduction paragraph 2"], ["Methods paragraph 1"]],
    #         "section_name": ["Introduction ::: Background ::: Details", "Methods ::: Experiment ::: Steps"]
    #     },
    # }

    # full_papers = {}
    process_all: bool = True
    case_study_only: List[str] = ["1709.06136"]
    # get all paper contents from test set
    for paper in raw_test:
        paper_id = paper["id"]
        paper_title = paper["title"]
        paragraphs = paper["full_text"]["paragraphs"]
        section_names = paper["full_text"]["section_name"]

        # full_papers[paper_id] = {}

        paper_texts: List[str] = []
        for section_idx, paras_in_section in enumerate(paragraphs):
            section_title = section_names[section_idx]
            section_text = f"\n{section_title}\n" + "\n".join(paras_in_section).strip()
            paper_texts.append(section_text)
        # full_papers[paper_id]["text"] = "".join(paper_texts)

        # Save processed data to TXT
        output_file = output_dir / f"{paper_id}.txt"
        with open(output_file, "w+", encoding="utf-8") as f:
            for text in paper_texts:
                f.write(text)
