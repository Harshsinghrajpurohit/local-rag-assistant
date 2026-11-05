# RAG Evaluation Report

- Questions: 47 (39 answerable, 8 unanswerable)
- Answer model: qwen3:4b-instruct | Judge: qwen3:4b-instruct

| Metric | Result |
|---|---|
| Recall@5 (source passage in top 5) | 85% (33/39) |
| MRR | 0.58 |
| Answer accuracy (CORRECT) | 85% (33/39) |
| Accuracy, PARTIAL counted as half | 88% |
| Accuracy when the source was retrieved | 91% |
| Exact-text match (no judge) | 79% |
| Correct refusals (unanswerable) | 100% (8/8) |
| Latency avg / p95 / max | 13.9s / 23.8s / 48.3s |
| Judge errors | 0 |

## Per document

| Document | Questions | Correct | Source retrieved |
|---|---|---|---|
| Attention_is_all_you_need.pdf | 5 | 100% | 80% |
| BAI702-module-1-textbook.pdf | 2 | 50% | 50% |
| BERT_paper.pdf | 8 | 88% | 88% |
| DPDT_11082024.pdf | 10 | 80% | 90% |
| Llama2_paper.pdf | 6 | 100% | 83% |
| RAG_paper.pdf | 8 | 75% | 88% |

## Items to review (6)

**Q001** [WRONG] What is the task T for handwriting recognition?
- Expected: recognizing and classifying handwritten words within images
- App said: 
- Judge: app error: LLM service temporarily unavailable

**Q018** [WRONG] What does "personal data" mean?
- Expected: any data about an individual who is identifiable by or in relation to such data
- App said: "Personal data" means a wholly or partly automated operation or set of operations performed on digital personal data, and includes operations such as collection, recording, organisation, structuring, storage, adaptation, retrieval, use, alignment or combination, indexing, sharing, disclosure by tran
- Judge: The system answer defines 'personal data' in terms of automated operations, which is not supported by the source passage.

**Q040** [PARTIAL] What is the role of a Data Fiduciary?
- Expected: be responsible for complying with the provisions of this Act
- App said: The role of a Data Fiduciary includes:

- Publishing the business contact information of a Data Protection Officer or a designated person who can address Data Principals' questions about personal data processing (Source 1).  
- Establishing an effective mechanism to redress the grievances of Data Pr
- Judge: The system answer includes additional responsibilities not supported by the source passage, such as grievance redress mechanisms, data erasure requests, and security safeguards, which are not mentione

**Q058** [PARTIAL] What year was the Cloze task introduced?
- Expected: 1953
- App said: The provided sources do not specify the year the Cloze task was introduced. While they mention the Cloze task (Taylor, 1953), they do not state the year of its introduction. Therefore, based on the information given, I do not have enough information to determine the year the Cloze task was introduce
- Judge: The system correctly identifies the reference (Taylor, 1953) but fails to state that the Cloze task was introduced in 1953, which is the key fact from the reference answer.

**Q082** [WRONG] What is the percentage gap in FEVER fact verification?
- Expected: 4.3%
- App said: The provided sources do not contain information about the percentage gap in FEVER fact verification. Therefore, I do not have enough information to answer the question.
- Judge: The system answer contradicts the source by stating it lacks information, while the source clearly states a 4.3% gap in FEVER fact verification.

**Q084** [PARTIAL] How many parameters does BART-large have?
- Expected: 400M
- App said: BART-large has 406M parameters.

Source 1 states: "406M trainable parameters from BART-large, 406M parameters"  
Source 4 states: "BART-large [32], a pre-trained seq2seq transformer [58] with 400M parameters."

There is a conflicting value: Source 1 reports 406M, while Source 4 reports 400M.

Theref
- Judge: The source passage explicitly states '400M parameters' and contradicts the system answer of '406M parameters'.
