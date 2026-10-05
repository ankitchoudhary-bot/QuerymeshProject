from fastmcp import Client


class MCPAgentClient:
    def __init__(self, url: str):
        self.url = url

    async def call(
        self,
        tool_name: str,
        arguments: dict,
    ):
        async with Client(self.url) as client:
            return await client.call_tool(
                tool_name,
                arguments,
            )
researcher = MCPAgentClient(
    "http://127.0.0.1:8001/sse"
)

retriever = MCPAgentClient(
    "http://127.0.0.1:8002/sse"
)

grader = MCPAgentClient(
    "http://127.0.0.1:8003/sse"
)      
            


async with client:
    result = await client.call_tool(
        "search_papers",
        {
            "query": "transformer attention mechanism"
        },
    )
    
    
    
class ResearcherRemoteClient:
    def __init__(self, url: str):
        self.url = url

    async def plan(self, query: str) -> dict:
        async with Client(self.url) as client:
            result = await client.call_tool(
                "search_papers",
                {"query": query},
            )

            return result
        
plan = await self.researcher_client.plan(query)



class RemoteAgentHandler:
    def __init__(self, client, tool_name):
        self.client = client
        self.tool_name = tool_name

    async def handle(self, message):
        return await self.client.call(
            self.tool_name,
            message.payload,
        )
        
handlers = {
    "researcher": RemoteAgentHandler(
        MCPAgentClient("http://127.0.0.1:8001/sse"),
        "search_papers",
    ),

    "retriever": RemoteAgentHandler(
        MCPAgentClient("http://127.0.0.1:8002/sse"),
        "retrieve_documents",
    ),

    "grader": RemoteAgentHandler(
        MCPAgentClient("http://127.0.0.1:8003/sse"),
        "grade_documents",
    ),
}