from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re


def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r"\s+", " ", query).strip()

    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r"\s*".join(re.escape(word) for word in normalized_query.split())

    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start() : match.end()], match.start(), match.end()
    else:
        return None


def rigorous_document_search(document: str, target: str):
    if target.endswith("."):
        target = target[:-1]
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r"[.!?]\s*|\n", document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)
    if best_match[1] < 98:
        return None
    reference = best_match[0]
    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index
