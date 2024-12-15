from datasets import load_dataset
from pathlib import Path
import json
from transformers import AutoTokenizer

# from sentence_transformers import SentenceTransformer
import torch
import re
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    raw_test = load_dataset(
        "allenai/qasper",
        split="test",
        cache_dir="/workspace/P76125041/.cache/huggingface/",
    )

    # Prepare LLM for summarization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        download_dir="/workspace/P76125041/models",
    )
    sampling_params = SamplingParams(min_tokens=500, max_tokens=800)

    # Create a Path object for the output directory
    output_dir = Path(f"qasper/")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize BERT summarization pipeline
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

    # Process dataset
    processed_dataset = {}
    llm_output = {}

    for paper in raw_test:
        paper_id = paper["id"]
        paper_title = paper["title"]
        paper_abstract = paper["abstract"]
        paragraphs = paper["full_text"]["paragraphs"]
        section_names = paper["full_text"]["section_name"]

        processed_dataset[paper_id] = {}
        llm_output[paper_id] = {}
        paragraph_id = 1

        for section_idx, paras_in_section in enumerate(paragraphs):
            # to_summarize: str = (
            #     f"Paper Title: {paper['title']}\n\nSection Title: {section_names[section_idx]}\n\nSection Text: {''.join(paras_in_section)}"
            # )
            # Generate section description using BERT summarizer
            # section_description = summarizer(to_summarize, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]

            section_title = section_names[section_idx]
            llm_output[paper_id][section_title] = {}

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in NLP research field.",
                },
                {
                    "role": "user",
                    # "content": f"Provided with the following inputs: (1) Title of a research paper in NLP; (2) Abstract summarizing the paper's content; (3) Table of contents outlining the structure of the paper. The objective is to produce an objective description of the content within the specified section, leveraging relevant knowledge in the field.\n\n(1) Title: {title}\n\n(2) Abstract: {abstract}\n\n(3) Table of contents: {sections}\n\nThe description for the section titled '{section_title}':",
                    "content": f"Provided with the following inputs: (1) Title of a research paper in NLP; (2) Abstract summarizing the paper's content; (3) Table of contents outlining the structure of the paper. The objective is to produce an objective description of the content within the specified section, leveraging relevant knowledge in the field.\n\n(1) Title: {paper_title}\n\n(2) Abstract: {paper_abstract}\n\n(3) Table of contents: {str(section_names)}\n\nThe description for the section titled '{section_title}' should be enclosed with '<<<' and '>>>':\n\n<<<\n",
                },
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outputs = model.generate(formatted_prompt, sampling_params)
            for output in outputs:
                llm_output[paper_id][section_title]["prompt"] = output.prompt
                generated_text = output.outputs[0].text
                llm_output[paper_id][section_title]["generated_text"] = generated_text
                match = re.search(r"<<<(.*?)>>>", generated_text, re.DOTALL)

                # Extract the match if found
                if match:
                    section_description = match.group(1).strip()
                    # print(extracted_text)
                else:
                    print("No match found.")
                    section_description = (
                        f"Paper Title: {paper_title}\n\nSection Title: {section_title}"
                    )
                break
                # print(f"Prompt: {prompt!r}\n\nGenerated text: {generated_text!r}")

            for paragraph in paras_in_section:
                processed_dataset[paper_id][paragraph_id] = {
                    "section_name": section_names[section_idx],
                    "section_description": section_description,
                    # "section_description": f"(wc: {len(to_summarize)}) {to_summarize[:100]}",
                    "text": paragraph,
                }
                paragraph_id += 1
    # save raw test dataset to JSON
    output_file = output_dir / "test_gold.json"
    with open(output_file, "w+", encoding="utf-8") as f:
        for paper in raw_test:
            f.write(json.dumps(paper, ensure_ascii=False))
            f.write("\n")

    # Save processed data to JSON
    output_file = output_dir / "test_papers.json"
    with open(output_file, "w+", encoding="utf-8") as f:
        json.dump(processed_dataset, f, indent=4)

    llm_output_file = output_dir / "llama3_8b_instruct_outputs.json"
    with open(llm_output_file, "w+", encoding="utf-8") as f:
        json.dump(llm_output, f, indent=4)

    # print(len(processed_dataset["1601.03313"]))
    # print(processed_dataset["1601.03313"][1])
    # print(f"Processed data saved to {output_file}")
