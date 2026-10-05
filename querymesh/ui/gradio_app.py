"""Gradio UI for QueryMesh.

The only module in this project that imports `gradio`. All actual work
(running the graph, streaming status) lives in `querymesh.ui.handlers`
and `querymesh.graph` — this file is purely widget wiring.
"""

from __future__ import annotations

import gradio as gr
from langgraph.graph.state import CompiledStateGraph

from querymesh.ui.handlers import run_query_streaming


def build_app(compiled: CompiledStateGraph, max_retries: int = 2) -> gr.Blocks:
    with gr.Blocks(title="QueryMesh", analytics_enabled=False) as demo:
        gr.Markdown("# QueryMesh — arXiv Research Assistant")
        gr.Markdown(
            "Ask a question about scientific research on arXiv. QueryMesh plans "
            "searches, retrieves and grades evidence, and synthesizes a grounded, "
            "cited answer."
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Research question",
                placeholder="e.g. What are the limitations of Retrieval-Augmented Generation?",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        status_box = gr.Textbox(label="Execution status", interactive=False)
        answer_box = gr.Markdown(label="Answer")
        figure_box = gr.Image(label="Retrieved Figure", type="filepath", )
        
        gr.Markdown("## Citations")

        sources_box = gr.HTML(value="<p>No citations yet.</p>",)

        gr.Markdown("## Retrieved Papers")

        papers_box = gr.JSON(label="Retrieved papers", )


        # with gr.Row():
        #     sources_box = gr.HTML(label="Citations")   # JSOn
        #     papers_box = gr.JSON(label="Retrieved papers")
            

        async def on_search(query: str):
            if not query or not query.strip():
                yield "Please enter a question.", "", "<p>No citations yet.</p>", [], None
                return

            async for update in run_query_streaming(compiled, query, max_retries=max_retries):
                yield update.status, update.answer, update.citations_html, update.retrieved_papers, update.figure

        search_btn.click(
            on_search,
            inputs=[query_input],
            outputs=[status_box, answer_box, sources_box, papers_box, figure_box],
        )
        query_input.submit(
            on_search,
            inputs=[query_input],
            outputs=[status_box, answer_box, sources_box, papers_box, figure_box],
        )

    return demo


def main() -> None:  # pragma: no cover - manual entrypoint, not unit tested
    from querymesh.embeddings import get_embedder
    from querymesh.graph import compile_workflow
    from querymesh.indexing import OpenSearchConnection, build_client
    from querymesh.llm import get_llm
    from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever

    llm = get_llm()
    client = build_client()
    connection = OpenSearchConnection(client=client, index_name="querymesh-papers")
    embedder = get_embedder()

    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(connection),
            vector_retriever=VectorRetriever(connection),
            embedder=embedder,
        )
    )

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
    app = build_app(compiled)
    app.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
