
# # Main app tests (tasks.py, no airflow needed):
# # pytest tests/unit/test_airflow_tasks.py -v

# # export PYTHONPATH=/path/to/querymesh
# # airflow dags list  # should show querymesh_arxiv_ingestion

# export PYTHONPATH="$PWD"
# export AIRFLOW__CORE__DAGS_FOLDER="$PWD/querymesh/airflow/dags"
# export AIRFLOW__CORE__LOAD_EXAMPLES=False


# wsl -d ubuntu
# cd /mnt/c/Users/91966/Downloads/QuerymeshProject/querymesh
# source .venv-airflow/bin/activate

#airflow scheduler
#airflow dags trigger querymesh_arxiv_ingestion
#airflow dags list-runs -d querymesh_arxiv_ingestion

# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" "https://localhost:9200/querymesh-papers/_count?pretty"

# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" "https://localhost:9200/querymesh-papers/_mapping?pretty"

#curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" "https://localhost:9200/querymesh-papers/_search?pretty&size=10&sort=chunk_id:asc" -H "Content-Type: application/json" -d '{\"query\":{\"term\":{\"arxiv_id\":\"2012.10055v2\"}},\"_source\":[\"arxiv_id\",\"title\",\"chunk_id\",\"text\"]}'


# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" `                                                                                 
# >>   "https://localhost:9200/querymesh-papers/_search?pretty&size=1" `                   
# >>   -H "Content-Type: application/json" `                                                     
# >>   -d '{\"query\":{\"term\":{\"chunk_id\":\"2012.10055v2:chunk:0000\"}},\"_source\":[\"arxiv_id\",\"title\",\"chunk_id\",\"text\"]}'
                                                        

# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" `                                                                                 
# >>   "https://localhost:9200/querymesh-papers/_search?pretty&size=10&_source=arxiv_id,chunk_id,text"


# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" `                                                                                 
# >>   "https://localhost:9200/querymesh-papers/_search?pretty&size=1" `                              
# >>   -H "Content-Type: application/json" `                                                     
# >>   -d '{\"query\":{\"term\":{\"chunk_id\":\"2012.10055v2:chunk:0000\"}},\"_source\":[\"arxiv_id\",\"title\",\"chunk_id\",\"text\",\"section\",\"page\"]}'     


# curl.exe -k -u "admin:V7#kP9!xR4@Lm2Q" `                                                                                 
# >>   "https://localhost:9200/querymesh-papers/_search?pretty&size=100" `                            
# >>   -H "Content-Type: application/json"                                             

# curl -k -u "admin:V7#kP9!xR4@Lm2Q" \
#   -X GET "https://localhost:9200/querymesh-papers/_search?pretty" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "size": 3,
#     "_source": [
#       "chunk_id",
#       "arxiv_id",
#       "chunk_type",
#       "page",
#       "text",
#       "image_path"
#     ],
#     "query": {
#       "term": {
#         "chunk_type": "figure"
#       }
#     }
#   }'

# set +H
          
# curl -k -u "admin:V7#kP9!xR4@Lm2Q" \
#   -X GET "https://localhost:9200/querymesh-papers/_search?pretty" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "size": 3,
#     "_source": {
#       "excludes": ["embedding"]
#     },
#     "query": {
#       "term": {
#         "chunk_type": "table"
#       }
#     }
#   }'                                                             
# from querymesh.ui import main
# main()  # launches Gradio at http://127.0.0.1:7860


from querymesh.ui import main
main()  # launches Gradio at http://127.0.0.1:7860


        
# from querymesh.security.mcp_auth import StaticBearerTokenVerifier
# from querymesh.mcp.servers import build_research_server

# verifier = StaticBearerTokenVerifier(settings.mcp_auth_token)  # from MCP_AUTH_TOKEN env var
# server = build_research_server(researcher, auth=verifier)

from querymesh.observability import get_tracer
from querymesh.graph import compile_workflow, create_initial_state, run_research_workflow
from querymesh.llm import get_llm
import asyncio

llm = get_llm()

from querymesh.indexing import build_client, OpenSearchConnection
from querymesh.embeddings import get_embedder
from querymesh.retrieval import BM25Retriever, VectorRetriever, HybridRetriever, RetrievalService

client = build_client()
conn = OpenSearchConnection(client=client, index_name="querymesh-papers")
from querymesh.graph import compile_workflow, create_initial_state, run_research_workflow_streaming
service = RetrievalService(
    hybrid_retriever=HybridRetriever(
        bm25_retriever=BM25Retriever(conn),
        vector_retriever=VectorRetriever(conn),
        embedder=get_embedder(),
    )
)

async def main():
    
    tracer = get_tracer()  # NoOp unless LOGFIRE_TOKEN / LANGFUSE_* are set in .env
    compiled = compile_workflow(llm=llm, retrieval_service=service, tracer=tracer)
    state = create_initial_state("what was dataset to train and test diarization models?", max_retries=2)
    async for event in run_research_workflow_streaming(compiled, state):
        print(event["type"], event["agent"], event["message"])

# asyncio.run(main())




import asyncio

from querymesh.llm.ollama import OllamaLLM


async def main():
    llm = OllamaLLM(model="llama3.2")

    print("Starting Ollama streaming...")

    chunks = []

    async for chunk in llm.astream(
        "Explain the transformer attention mechanism in 3 sentences."
    ):
        print(
            f"chunk={chunk.text!r}, "
            f"final={chunk.is_final}"
        )

        chunks.append(chunk.text)

    print("\nFINAL:")
    print("".join(chunks))


# asyncio.run(main())




import asyncio

from querymesh.ui.handlers import run_query_streaming
# construct compiled exactly as run.py does

compiled = compile_workflow(llm=llm, retrieval_service=service, )
async def test():
    async for update in run_query_streaming(
        compiled,
        "What are the limitations of RAG?",
    ):
        print(
            "UI UPDATE:",
            repr(update.answer[-50:]),
            update.status,
        )


# asyncio.run(test())


# from querymesh.ingestion import PdfDownloader, IngestionPipeline, DoclingParser
# from querymesh.ingestion.models import PaperMetadata
# import asyncio

# async def main():
#     downloader = PdfDownloader(storage_dir="./data/pdfs")
#     pipeline = IngestionPipeline(downloader=downloader, parser=DoclingParser())
#     paper = PaperMetadata(arxiv_id="2608.13558v1", title="...", abstract="...",
#                            pdf_url="https://arxiv.org/pdf/2608.13558v1")
#     doc = await pipeline.process(paper)

#     print("\n========== SECTIONS ==========")
#     print("Sections:", len(doc.sections))

#     print("\n========== TABLES ==========")
#     print("Tables:", len(doc.tables))

#     for i, table in enumerate(doc.tables, 1):
#         print(f"\nTable {i}")
#         print("Caption:", table.caption)
#         print("Page:", table.page)
#         print("Image:", table.image_path)
#         print("Markdown:")
#         print(table.markdown)

#     print("\n========== FIGURES ==========")
#     print("Figures:", len(doc.figures))

#     for i, figure in enumerate(doc.figures, 1):
#         print(f"\nFigure {i}")
#         print("Caption:", figure.caption)
#         print("Page:", figure.page)
#         print("Image:", figure.image_path)

# asyncio.run(main())




# from pathlib import Path
# import asyncio

# from querymesh.vision.ollama import OllamaVisionModel


# vision = OllamaVisionModel(
#     model="qwen3-vl:4b",
# )


# async def test_vision():
#     description = await vision.describe_image(
#         Path(
#             "data/images/2608.13558v1/"
#             "2608.13558v1_figure_14.png"
#         ),
#         context=None,
#     )

#     print("\n========== FIGURE DESCRIPTION ==========\n")
#     print(description)
#     print("\n========================================\n")


# asyncio.run(test_vision())

# # from pathlib import Path
# # from querymesh.vision.ollama import FigureVisionDescriber

# # vision = FigureVisionDescriber(
# #     model="qwen3-vl:4b",
# # )
# # async def main():
  
# #   description = await vision.describe(
# #       Path("data/images/2012.10055v2/2012.10055v2_figure_1.png"),
# #       caption="Fig. 1: The flow of the proposed method...",
# #   )

# #   print(description)
  
  
# # asyncio.run(main())





import asyncio

from querymesh.embeddings import get_embedder
from querymesh.indexing import OpenSearchConnection, build_client
from querymesh.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RetrievalService,
    VectorRetriever,
)


# async def main():
#     client = build_client()

connection = OpenSearchConnection(
        client=client,
        index_name="querymesh-papers",
)

embedder = get_embedder()

retrieval_service = RetrievalService(
    hybrid_retriever=HybridRetriever(
        bm25_retriever=BM25Retriever(connection),
        vector_retriever=VectorRetriever(connection),
        embedder=embedder,
    )
)

    # try:
    #     results = await retrieval_service.retrieve(
    #         "flow of the proposed method when the number of speakers is three",
    #         top_k=10,
    #     )

    #     for i, result in enumerate(results, start=1):
    #         doc = result.document

    #         print("\n" + "=" * 80)
    #         print(f"RESULT {i}")
    #         print("=" * 80)
    #         print("chunk_id   :", doc.get("chunk_id"))
    #         print("chunk_type :", doc.get("chunk_type"))
    #         print("arxiv_id   :", doc.get("arxiv_id"))
    #         print("title      :", doc.get("title"))
    #         print("page       :", doc.get("page"))
    #         print("image_path :", doc.get("image_path"))
    #         print("score      :", result.score)
    #         print("method     :", result.retrieval_method)

    # finally:
    #     await connection.close()


# if __name__ == "__main__":
#     asyncio.run(main())


# from querymesh.graph import compile_workflow, create_initial_state
# from querymesh.llm import get_llm
# async def main():
#     compiled = compile_workflow(llm=get_llm(), retrieval_service=retrieval_service)
#     result = await compiled.ainvoke(create_initial_state("What are the limitations of RAG?", max_retries=1))
#     print(result["search_queries"])  # different on each retry now, not resent
#     # print(result["grading_reasons"])
#     print(result["retry_count"])
    
    
# if __name__ == "__main__":
#     asyncio.run(main())


from querymesh.cache import get_response_cache
from querymesh.observability import get_tracer
from querymesh.graph import compile_workflow, create_initial_state, run_research_workflow, run_research_workflow_streaming
tracer=get_tracer()
cache = get_response_cache(tracer=tracer)
compiled = compile_workflow(llm=get_llm(), retrieval_service=retrieval_service)

async def sexy():

    # result = await run_research_workflow(compiled=compiled, initial_state=create_initial_state("What are the limitations of RAG?"), tracer=tracer,cache=cache, )
    # # cached = await cache.get("What are the limitations of RAG?")
    # # if cached is None:
    # #     # ... run the graph, get answer + citations ...
    # #     await cache.set("What are the limitations of RAG?", answer="redis caching is working", citations="I am saying na so it's true only!")
    
    # print(result["answer"])
    # print(result["cache_hit"])
    # print(result["citations"])
        
    async for event in run_research_workflow_streaming(
        compiled=compiled,
        initial_state=create_initial_state("What are the limitations of RAG?"),
        tracer=tracer,
        cache=cache,
    ):
        
        # print(event["answer"])
        # print(event["cache_hit"])
        # print(event["citations"])
        # print(event["metadata"])
        print(event)
    

# if __name__== "__main__":
#     asyncio.run(sexy())
    
    
    
# async def test_workflow_uses_response_cache():
#     cache = get_response_cache(tracer=tracer)

#     first = await run_research_workflow_streaming(
#         compiled=compiled,
#         initial_state = create_initial_state("What are the limitations of RAG?"),
#         cache=cache,
#         tracer=tracer
#     )

#     second = await run_research_workflow_streaming(
#         compiled=compiled,
#         initial_state= create_initial_state("What are the limitations of RAG?"),
#         cache=cache,
#         tracer=tracer
#     )

#     assert first["answer"] == second["answer"]

