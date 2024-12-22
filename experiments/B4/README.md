# TODOs
- generate section description using LLama3
- 理解如何使用 LED-base (input & output format?) 
- 理解如何用 fine-tuned baseline LED (input & output format?) 
- complete reader module, which outputs answer as string, and save it in to `test_predictions.jsonl`
- try running `evaluator.py`, define where to store results
- implement different experimental settings

## Prepare DB
- Download QASPER test split (save as gold DB)
- papers DB: 

paper ID -> para ID -> `section_name`; `section_description`; `paragraph_text`
- questions DB:

question ID -> `question_text`; `paper_source`; `answers`

## Experimental Settings
### fixed
- LLM for generating section descriptions: `meta-llama/Meta-Llama-3-8B-Instruct`
- embedding model for para_text embeddings & section_description embeddings: `dunzhang/stella_en_400M_v5`
    - why? top ranked in MTEB retrieval & reranking task

1. read DB
2. convert para embeddings
3. get top-k
4. convert section embeddings of top-k
5. get reranked top-k
6. update predictions dict(evidences)
7. format as LED input
8. update predictions dict(answers)
9. run eval (save results to specified output dir)

### observation 1: 標題的重要性
- 無 rerank，不需要 section description
- 建立 3 種 para_text：
    - no title, pure para_text
    - top-level title as prefix (format - `Title: {title}\n\nText: {para_text}`)
    - full headings as prefix (format - `Title: {full-title}\n\nText: {para_text}`)
- retrieved top-5 作爲 led-base 的 input，得到 answer
- 記錄 answer F1 vs. evidence F1

### observation 2: rerank的重要性
- 建立 3 種 para_text：
    - no title, pure para_text
    - top-level title as prefix (format - `Title: {title}\n\nText: {para_text}`)
    - full headings as prefix (format - `Title: {full-title}\n\nText: {para_text}`)
- retrived top-k 後，再用 section_description 轉成的 embedding 算相似度，重新排名，取 top-5
- reranked top-5 作爲 led-base 的 input，得到 answer
- 記錄 answer F1 vs. evidence F1





