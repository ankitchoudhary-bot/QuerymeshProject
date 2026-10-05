# querymesh/services/retriever_service.py

from querymesh.agents.retriever import RetrieverAgent
from querymesh.mcp.servers.retrieval_server import build_retrieval_server


def create_retriever() -> RetrieverAgent:
    # construct your OpenSearch / embedding dependencies here
    return RetrieverAgent(...)


if __name__ == "__main__":
    retriever = create_retriever()

    server = build_retrieval_server(retriever)

    server.run(
        transport="sse",
        host="127.0.0.1",
        port=8002,
    )
    
    