from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        download_dir="/workspace/P76125041/models",
    )
    sampling_params = SamplingParams(min_tokens=500, max_tokens=800)
    title = "End-to-End Trainable Non-Collaborative Dialog System"
    abstract = "End-to-end task-oriented dialog models have achieved promising performance on collaborative tasks where users willingly coordinate with the system to complete a given task. While in non-collaborative settings, for example, negotiation and persuasion, users and systems do not share a common goal. As a result, compared to collaborate tasks, people use social content to build rapport and trust in these non-collaborative settings in order to advance their goals. To handle social content, we introduce a hierarchical intent annotation scheme, which can be generalized to different non-collaborative dialog tasks. Building upon TransferTransfo (Wolf et al. 2019), we propose an end-to-end neural network model to generate diverse coherent responses. Our model utilizes intent and semantic slots as the intermediate sentence representation to guide the generation process. In addition, we design a filter to select appropriate responses based on whether these intermediate representations fit the designed task and conversation constraints. Our non-collaborative dialog model guides users to complete the task while simultaneously keeps them engaged. We test our approach on our newly proposed ANTISCAM dataset and an existing PERSUASIONFORGOOD dataset. Both automatic and human evaluations suggest that our model outperforms multiple baselines in these two non-collaborative tasks."
    sections = '["Introduction", "Related Work", "Non-Collaborative Task Annotation Scheme", "Datasets", "Datasets ::: AntiScam Dataset", "Datasets ::: PersuasionForGood Dataset", "Model ::: Background", "Model ::: Intent and Semantic Slot Classifiers", "Model ::: Response Generation", "Model ::: Response Filtering", "Experiments", "Experiments ::: Baseline Models", "Experiments ::: Automatic Evaluation Metrics", "Experiments ::: Human Evaluation Metrics", "Results and Analysis", "Conclusion and Future Work", "Acknowledgements", "Appendix ::: Anti-Scam Collection Setting", "Appendix ::: Training details", "Appendix ::: Example Dialog"]'
    section_title = "Experiments ::: Automatic Evaluation Metrics"
    messages = [
        {
            "role": "system",
            "content": "You are an expert in NLP research field.",
        },
        {
            "role": "user",
            # "content": f"Provided with the following inputs: (1) Title of a research paper in NLP; (2) Abstract summarizing the paper's content; (3) Table of contents outlining the structure of the paper. The objective is to produce an objective description of the content within the specified section, leveraging relevant knowledge in the field.\n\n(1) Title: {title}\n\n(2) Abstract: {abstract}\n\n(3) Table of contents: {sections}\n\nThe description for the section titled '{section_title}':",
            "content": f"Provided with the following inputs: (1) Title of a research paper in NLP; (2) Abstract summarizing the paper's content; (3) Table of contents outlining the structure of the paper. The objective is to produce an objective description of the content within the specified section, leveraging relevant knowledge in the field.\n\n(1) Title: {title}\n\n(2) Abstract: {abstract}\n\n(3) Table of contents: {sections}\n\nThe description for the section titled '{section_title}' should be enclosed with '<<<' and '>>>':\n\n<<<\n",
        },
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = model.generate(formatted_prompt, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\n\nGenerated text: {generated_text!r}")
