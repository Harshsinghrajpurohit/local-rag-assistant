# RAG Evaluation Report

- Questions: 20 (12 answerable, 8 unanswerable)
- Answer model: qwen3:4b-instruct | Judge: qwen3:4b-instruct

| Metric | Result |
|---|---|
| Recall@5 (source passage in top 5) | 75% (9/12) |
| MRR | 0.46 |
| Answer accuracy (CORRECT) | 75% (9/12) |
| Accuracy, PARTIAL counted as half | 79% |
| Accuracy when the source was retrieved | 89% |
| Exact-text match (no judge) | 67% |
| Correct refusals (unanswerable) | 100% (8/8) |
| Latency avg / p95 / max | 11.2s / 26.2s / 27.7s |
| Judge errors | 0 |

## Per document

| Document | Questions | Correct | Source retrieved |
|---|---|---|---|
| BAI702-module-1-textbook.pdf | 2 | 100% | 100% |
| DPDT_11082024.pdf | 10 | 70% | 70% |

## Items to review (3)

**Q018** [WRONG] What does "personal data" mean?
- Expected: any data about an individual who is identifiable by or in relation to such data
- App said: "Personal data" means digital personal data that is the subject of processing, including operations such as collection, recording, organisation, structuring, storage, adaptation, retrieval, use, alignment or combination, indexing, sharing, disclosure by transmission, and any other operations perform
- Judge: The system answer introduces unsupported details like 'digital personal data' and references a non-existent 'Digital Personal Data Protection Act, 2023', which are not in the source passage.

**Q033** [WRONG] What does "automated" mean in this Act?
- Expected: any digital process capable of operating automatically
- App said: The provided sources do not define "automated" in the context of this Act. While Source 1 mentions "automated operation or set of operations" in relation to processing of personal data, it does not explain what "automated" means. Therefore, based on the provided documents, there is no explicit defin
- Judge: The system answer contradicts the source by stating that the sources do not define 'automated,' when the source clearly defines it as 'any digital process capable of operating automatically'.

**Q040** [PARTIAL] What is the role of a Data Fiduciary?
- Expected: be responsible for complying with the provisions of this Act
- App said: The role of a Data Fiduciary includes:

- Ensuring compliance with the provisions of the Act and rules made thereunder, regardless of any agreement or failure by a Data Principal to fulfill duties (Source 3).  
- Establishing an effective mechanism to redress the grievances of Data Principals (Sourc
- Judge: The system answer includes multiple responsibilities not supported by the source passage, while the reference answer correctly identifies compliance as the primary role.
