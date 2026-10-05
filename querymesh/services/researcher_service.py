from querymesh.agents.researcher import ResearcherAgent
from querymesh.llm.ollama import OllamaLLM
from querymesh.mcp.servers.research_server import build_research_server


def create_researcher() -> ResearcherAgent:
    llm = OllamaLLM(
        model="llama3.2",
        base_url="http://localhost:11434",
    )

    return ResearcherAgent(llm=llm)


if __name__ == "__main__":
    researcher = create_researcher()

    server = build_research_server(researcher)

    server.run(
        transport="sse",
        host="127.0.0.1",
        port=8001,
    )