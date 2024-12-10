from chromadb import Documents, EmbeddingFunction, Embeddings
from chunking_evaluation.utils import openai_token_count
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import ast
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
HF_READ = os.getenv("HF_READ")


# Define a custom chunking class
class PropositionizerChunker(BaseChunker):
    def __init__(
        self, model_name: str = "chentong00/propositionizer-wiki-flan-t5-large"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.splitter = RecursiveTokenChunker(
            chunk_size=100,
            chunk_overlap=0,
            length_function=openai_token_count,
            # separators=[".", "!", "?"],
        )

    def split_text(self, text):
        passages = self.splitter.split_text(text)
        propositions = []
        for passage in passages:
            input_ids = self.tokenizer(passage, return_tensors="pt").input_ids
            outputs = self.model.generate(
                input_ids.to(self.device), max_new_tokens=512
            ).cpu()
            try:
                output_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
            except Exception as e:
                output_text = "[]"
            # print(output_text)
            output_text = output_text.strip()
            # if not output_text.endswith('"]'):
            #     output_text += '"]'
            #     print(output_text)
            # elif not output_text.endswith("]") and output_text.endswith('"'):
            #     output_text += "]"
            #     print(output_text)
            # if not output_text.startswith('["'):
            #     output_text = '["' + output_text
            #     print(output_text)
            # elif output_text.startswith("[") and output_text[1] != '"':
            #     output_text = '["' + output_text[1:]
            #     print(output_text)
            # propositions = [output_text]
            try:
                outputs = [o.strip() for o in ast.literal_eval(output_text)]
                # print(output_text)
                # outputs = [output_text]
                # print(isinstance(outputs, list))
            # except ValueError:
            #     outputs = []
            except SyntaxError:
                outputs = [output_text]
            except ValueError:
                continue
            if outputs and isinstance(outputs, list):
                print(outputs)
                propositions.extend(outputs)
        with open("final_propositions.txt", "a+") as file:
            file.write("\n")
            file.write("=" * 100)
            file.write("\n")
            for p in propositions:
                file.write(p + "\n")
        return propositions


if __name__ == "__main__":
    # Instantiate the custom chunker and evaluation
    chunker = PropositionizerChunker()
    evaluation = GeneralEvaluation()

    # Choose embedding function
    default_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_KEY, model_name="text-embedding-3-large"
    )

    # with open("wiki_raw.txt", "r") as file:
    #     lines = file.readlines()
    #     input_text = "\n".join(lines)

    # title = "Leaning Tower of Pisa"
    # section = ""
    # content = "Prior to restoration work performed between 1990 and 2001, Leaning Tower of Pisa leaned at an angle of 5.5 degrees, but the tower now leans at about 3.99 degrees. This means the top of the tower is displaced horizontally 3.9 meters (12 ft 10 in) from the center."

    # input_text = f"Title: {title}. Section: {section}. Content: {content}"

    # Evaluate the chunker
    results = evaluation.run(chunker, default_ef)
    # passages = chunker.splitter.split_text(input_text)
    # with open("wiki_passages.txt", "w+") as file:
    #     for p in passages:
    #         file.write(p)
    #         file.write("\n")
    #         file.write("+" * 100)
    #         file.write("\n")

    # with open("wiki_propositions.txt", "w+") as file:
    #     for passage in passages:
    #         pros = chunker.split_text(passage)
    #         for p in pros:
    #             file.write(p)
    #             file.write("\n")
    #             file.write("+" * 100)
    #             file.write("\n")
    #         file.write("=" * 100)

    print(results)
