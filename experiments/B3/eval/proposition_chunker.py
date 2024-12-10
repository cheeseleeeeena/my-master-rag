from chunking_evaluation import BaseChunker, GeneralEvaluation
from chunking_evaluation.utils import openai_token_count
from chunking_evaluation.chunking import RecursiveTokenChunker
from chromadb.utils import embedding_functions
import backoff
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
HF_READ = os.getenv("HF_READ")


class OpenAIClient:
    def __init__(self, model_name, api_key=None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            gpt_messages = [{"role": "system", "content": system_prompt}] + messages

            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=gpt_messages,
                temperature=temperature,
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e


class PropositionChunker(BaseChunker):
    """
    LLMSemanticChunker is a class designed to split text into thematically consistent sections based on suggestions from a Language Model (LLM).
    Users can choose between OpenAI and Anthropic as the LLM provider.

    Args:
        organisation (str): The LLM provider to use. Options are "openai" (default).
        api_key (str, optional): The API key for the chosen LLM provider. If not provided, the default environment key will be used.
        model_name (str, optional): The specific model to use. Defaults to "gpt-4o" for OpenAI.
                                    Users can specify a different model by providing this argument.
    """

    def __init__(
        self, organisation: str = "openai", api_key: str = None, model_name: str = None
    ):
        if organisation == "openai":
            if model_name is None:
                model_name = "gpt-4o"
            self.client = OpenAIClient(model_name, api_key=api_key)
        else:
            raise ValueError("Invalid organisation.")

        self.splitter = RecursiveTokenChunker(
            chunk_size=100, chunk_overlap=0, length_function=openai_token_count
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: "
                    + str(current_chunk)
                    + "."
                    + (
                        f"\n\The previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again."
                        if invalid_response
                        else ""
                    )
                ),
            },
        ]
        return messages

    def split_text(self, text):
        import re

        chunks = self.splitter.split_text(text)

        split_indices = []

        short_cut = len(split_indices) > 0

        from tqdm import tqdm

        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0

                chunked_input = ""

                for i in range(current_chunk, len(chunks)):
                    token_count += openai_token_count(chunks[i])
                    chunked_input += (
                        f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    )
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.client.create_message(
                        messages[0]["content"],
                        messages[1:],
                        max_tokens=200,
                        temperature=0.2,
                    )
                    # Use regular expression to find all numbers in the string
                    split_after_line = [
                        line
                        for line in result_string.split("\n")
                        if "split_after:" in line
                    ][0]
                    numbers = re.findall(r"\d+", split_after_line)
                    # Convert the found numbers to integers
                    numbers = list(map(int, numbers))

                    # print(numbers)

                    # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                    if not (
                        numbers != sorted(numbers)
                        or any(number < current_chunk for number in numbers)
                    ):
                        break
                    else:
                        messages = self.get_prompt(
                            chunked_input, current_chunk, numbers
                        )
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)

                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk = ""
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + " "
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs


if __name__ == "__main__":
    # Instantiate the custom chunker and evaluation
    chunker = PropositionChunker()
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

    # passages = chunker.splitter.split_text(input_text)
    # with open("wiki_passages.txt", "w+") as file:
    #     for p in passages:
    #         file.write(p)
    #         file.write("\n")
    #         file.write("+" * 100)
    #         file.write("\n")

    # pros = chunker.split_text(input_text)
    # with open("wiki_propositions.txt", "w+") as file:
    #     for p in pros:
    #         file.write(p)
    #         file.write("\n")
    #         file.write("+" * 100)
    #         file.write("\n")

    # Evaluate the chunker
    results = evaluation.run(chunker, default_ef)
    print(results)
