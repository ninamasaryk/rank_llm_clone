
from rank_llm.retrieve_and_rerank import retrieve_and_rerank
from rank_llm.data import Candidate, Query, Request

requests = [Request(query=Query(text=['dsaEFWEF'], qid=1), 
                            candidates=[Candidate(doc = {"text":'ABC',}, 
                                                  score=0.85, 
                                                  docid=0), Candidate(doc = {"text":'CNAPUH',}, 
                                                  score=0.55, 
                                                  docid=2)])]
result = retrieve_and_rerank(model_path='rank_zephyr',
                                    requests=[])