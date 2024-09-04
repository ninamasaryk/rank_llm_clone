import copy
from typing import Any, Dict, List, Union

from rank_llm.data import Query, Request
# from rank_llm.evaluation.trec_eval import EvalFunction
from rank_llm.rerank import IdentityReranker, RankLLM, Reranker
from rank_llm.rerank.reranker import extract_kwargs
from rank_llm.retrieve import (
    TOPICS,
    RetrievalMethod,
    RetrievalMode,
    Retriever,
    ServiceRetriever,
)


def retrieve_and_rerank(
    model_path: str,
    top_k_retrieve: int = 5,
    top_k_rerank: int = 5,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    requests: List[Request] = None,
    num_passes: int = 1,
    interactive: bool = False,
    default_agent: RankLLM = None,
    **kwargs: Any,
):
    """Retrieve candidates using Anserini API and rerank them

    Returns:
        - List of top_k_rerank candidates
    """


    # Get reranking agent
    reranker = Reranker(
        Reranker.create_agent(model_path.lower(), default_agent, interactive, num_gpus = 4, **kwargs)
    )

    # Reranking stage
    print(f"Reranking and returning {top_k_rerank} passages with {model_path}...")
    if reranker.get_agent() is None:
        # No reranker. IdentityReranker leaves retrieve candidate results as is or randomizes the order.
        shuffle_candidates = True if model_path == "rank_random" else False
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=(shuffle_candidates),
        )
    else:
        # Reranker is of type RankLLM
        for pass_ct in range(num_passes):
            print(f"Pass {pass_ct + 1} of {num_passes}:")

            rerank_results = reranker.rerank_batch(
                requests,
                rank_end=top_k_retrieve,
                rank_start=0,
                shuffle_candidates=shuffle_candidates,
                logging=print_prompts_responses,
                top_k_retrieve=top_k_retrieve,
                **kwargs,
            )

        if num_passes > 1:
            requests = [
                Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                for r in rerank_results
            ]

        for rr in rerank_results:
            rr.candidates = rr.candidates[:top_k_rerank]

    print(f"Reranking with {num_passes} passes complete!")

    if interactive:
        return (rerank_results, reranker.get_agent())
    else:
        return rerank_results
    
def retrieve(
    top_k_retrieve: int = 50,
    interactive: bool = False,
    retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
    retrieval_method: RetrievalMethod = RetrievalMethod.BM25,
    query: str = "",
    qid: int = 1,
    **kwargs,
):
    """Retrieve initial candidates

    Keyword arguments:
    dataset -- dataset to search if interactive
    top_k_retrieve -- top k candidates to retrieve
    retrieval_mode -- Mode of retrieval
    retrieval_method -- Method of retrieval
    query -- query to retrieve against
    qid - qid of query

    Return: requests -- List[Requests]
    """

    # Retrieve
    if interactive and retrieval_mode != RetrievalMode.DATASET:
        raise ValueError(
            f"Unsupport retrieval mode for interactive retrieval. Currently only DATASET mode is supported."
        )

    requests: List[Request] = []
    if retrieval_mode == RetrievalMode.DATASET:
        host: str = kwargs.get("host", "http://localhost:8081")
        dataset: Union[str, List[str], List[Dict[str, Any]]] = kwargs.get(
            "dataset", None
        )
        if dataset == None:
            raise ValueError("Must provide a dataset")

        if interactive:
            service_retriever = ServiceRetriever(
                retrieval_method=retrieval_method, retrieval_mode=retrieval_mode
            )

            # Calls Anserini API
            requests = [
                service_retriever.retrieve(
                    dataset=dataset,
                    request=Request(query=Query(text=query, qid=qid)),
                    k=top_k_retrieve,
                    host=host,
                )
            ]
        else:
            requests = Retriever.from_dataset_with_prebuilt_index(
                dataset_name=dataset,
                retrieval_method=retrieval_method,
                k=top_k_retrieve,
            )
    elif retrieval_mode == RetrievalMode.CUSTOM:
        keys_and_defaults = [
            ("index_path", None),
            ("topics_path", None),
            ("index_type", None),
        ]
        [index_path, topics_path, index_type] = extract_kwargs(
            keys_and_defaults, **kwargs
        )
        requests = Retriever.from_custom_index(
            index_path=index_path, topics_path=topics_path, index_type=index_type
        )

    return requests
