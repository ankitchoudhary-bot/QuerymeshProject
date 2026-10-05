from __future__ import annotations

import base64
from pathlib import Path
import httpx

from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

class OllamaVisionModel:
    """Vision model backed by a local Ollama server.
    Generate scientific figure descriptions using an Ollama vision model."""

    def __init__(self, model: str, base_url: str = "http://172.28.240.1:11434", timeout: float = 1200.0, ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def describe(self, image_path: Path, caption: str | None = None, ) -> str:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Figure image not found: {image_path}")

        image_bytes = path.read_bytes()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = self._build_prompt(caption)

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            "options": 
                {
                    "temperature": 0.0,
                },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:

            response = await client.post(f"{self.base_url}/api/chat", json=payload,)
            response.raise_for_status()
            data = response.json()

        return (data.get("message", {}).get("content", "").strip())

    @staticmethod
    def _build_prompt(context: str | None,) -> str:

        context_text = ""

        if context:
            context_text = f"""
            The figure comes from a scientific research paper.

            Figure caption:
            {context}
            """

        return f"""
            You are analyzing a figure from a scientific research paper.

            {context_text}

            Describe the figure for a retrieval-augmented generation system.

            Focus on:
            1. What the figure represents.
            2. The main components or entities.
            3. Relationships between components.
            4. Important trends, comparisons, or results.
            5. Labels, axes, legends, or annotations when readable.
            6. Any architectural or workflow relationships.
            7. Important scientific information that would help answer
            questions about this figure.

            Do not invent information that is not visible.

            Return a concise but information-dense description in plain text.
            Do not use phrases such as "I can see" or "the image shows".
            """.strip()
            

    
    # @staticmethod
    # def _build_prompt(context: str | None) -> str:

    #     context_text = ""

    #     if context:
    #         context_text = f"""
    # The figure comes from a scientific research paper.

    # Figure caption:
    # {context}
    # """

    #     return f"""
    # You are a scientific figure description generator for a
    # retrieval-augmented generation (RAG) system.

    # {context_text}

    # Include:
    # - what the figure represents
    # - the purpose of each panel
    # - important plots, diagrams, tables, or relationships
    # - readable labels, axes, legends, and annotations
    # - important numerical values
    # - important comparisons and trends
    # - conclusions directly supported by the figure

    # Use both the figure and the supplied caption when available.

    # Do not invent information.
    # Do not infer unstated scientific conclusions.
    # Do not discuss image quality or irrelevant visual styling.
    # Do not provide reasoning.
    # Return only the final description.
    # Keep the description concise and information-dense.""".strip()

    # Analyze the provided scientific figure and produce a single,
    # information-dense description that can be indexed and retrieved
    # to answer questions about the figure.

    # Describe only information supported by the figure and caption.

    # Include, when applicable:

    # 1. Figure purpose
    # - What scientific problem, experiment, method, or result is presented.

    # 2. Figure structure
    # - Number of panels and what each panel represents.

    # 3. Visual data
    # - Plots, diagrams, tables, charts, traces, or other visual elements.

    # 4. Important labels
    # - Axes, legends, annotations, categories, component names,
    #     and readable text.

    # 5. Numerical information
    # - Important values, percentages, measurements, thresholds,
    #     parameters, or ranges.

    # 6. Relationships and comparisons
    # - Differences between groups, trends, correlations,
    #     increases/decreases, or comparisons between methods.

    # 7. Scientific findings
    # - State conclusions that are directly supported by the figure.

    # For plots, include:
    # - x-axis
    # - y-axis
    # - important curves, bars, or traces
    # - notable values and trends

    # For diagrams or architectures, describe:
    # - major components
    # - connections
    # - direction of information or data flow

    # For tables, describe:
    # - columns and rows
    # - important values
    # - notable comparisons

    # Do not:
    # - invent information
    # - infer facts that cannot be supported by the figure
    # - discuss image quality
    # - describe irrelevant visual styling or colors
    # - provide chain-of-thought or reasoning
    # - say "I can see", "the image shows", or similar phrases

    # Return ONLY the final description in plain text.
    # Do not use numbered sections, markdown headings, or bullet points.
    # """.strip()
            
            
# You are analyzing a figure from a scientific research paper.

# Describe the figure accurately for use in a retrieval-augmented generation
# (RAG) system.

# Focus on:
# - what the figure represents
# - charts, plots, diagrams, architecture, or experimental results
# - axes and their labels
# - legends
# - important trends or comparisons
# - numbers that are clearly readable
# - relationships between components
# - visual structure and flow
# - any important annotations

# Do not invent information that is not visible.

# If some text or value is unreadable, say so.

# Paper caption:
# {caption}

# Return a concise but information-dense description.



# from __future__ import annotations

# import base64
# from pathlib import Path
# import httpx
# from querymesh.observability.logging import get_logger

# logger = get_logger(__name__)

# _DEFAULT_BASE_URL = "http://localhost:11434"
# _DEFAULT_MODEL = "qwen3-vl:4b"
# _DEFAULT_TIMEOUT_SECONDS = 3000.0

# class VisionModelError(RuntimeError):
#     """Raised when the Ollama vision model cannot process a figure."""

# class FigureVisionDescriber:
#     """
#     Generate scientific figure descriptions using an Ollama vision model.

#     The model receives:
#         - the figure image
#         - the original paper caption, when available

#     and returns an information-dense textual description suitable for
#     retrieval-augmented generation (RAG).
#     """

#     def __init__(
#         self,
#         model: str = _DEFAULT_MODEL,
#         base_url: str = _DEFAULT_BASE_URL,
#         timeout: float = _DEFAULT_TIMEOUT_SECONDS,
#     ) -> None:
#         self.model = model
#         self.base_url = base_url.rstrip("/")
#         self.timeout = timeout

#     async def describe(self, image_path: Path, caption: str | None = None, ) -> str:
#         """
#         Generate a description for a scientific figure.

#         Args:
#             image_path: Path to the figure image.
#             caption: Original figure caption extracted from the paper.

#         Returns:
#             A concise, information-dense figure description.

#         Raises:
#             VisionModelError:
#                 If the image cannot be read, Ollama cannot be reached,
#                 the request fails, or Ollama returns an empty response.
#         """

#         image_path = Path(image_path)

#         if not image_path.exists():
#             raise VisionModelError(f"Figure image does not exist: {image_path}")

#         if not image_path.is_file():
#             raise VisionModelError(f"Figure path is not a file: {image_path}")

#         try:
#             # ---------------------------------------------------------
#             # Read and encode image
#             # ---------------------------------------------------------
#             image_bytes = image_path.read_bytes()

#             if not image_bytes:
#                 raise VisionModelError(f"Figure image is empty: {image_path}")

#             image_b64 = base64.b64encode(image_bytes).decode("utf-8")

#             # ---------------------------------------------------------
#             # Build vision prompt
#             # ---------------------------------------------------------
#             prompt = self._build_prompt(caption)

#             # ---------------------------------------------------------
#             # Ollama multimodal chat request
#             # ---------------------------------------------------------
#             payload = {
#                 "model": self.model,
#                 "stream": False,
#                 "think": False,
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": prompt,
#                         "images": [image_b64],
#                     }
#                 ],
#                 "options": {
#                     "temperature": 0.0,
#                     "num_predict": 800,
#                 },
#             }

#             logger.info(
#                 "Generating figure description",
#                 extra={"image_path": str(image_path), "model": self.model, },
#             )

#             # ---------------------------------------------------------
#             # Call Ollama
#             # ---------------------------------------------------------
#             async with httpx.AsyncClient(timeout=self.timeout) as client:
#                 response = await client.post(f"{self.base_url}/api/chat", json=payload, )

#             response.raise_for_status()

#             # ---------------------------------------------------------
#             # Parse response
#             # ---------------------------------------------------------
#             try:
#                 data = response.json()
                
#             except ValueError as exc:
#                 raise VisionModelError("Ollama returned an invalid JSON response") from exc
            
#             print("========== RAW OLLAMA RESPONSE ==========")
#             print(data)

#             description = (data.get("message", {}).get("content", "").strip())

#             if not description:
#                 raise VisionModelError(f"Ollama returned an empty description for {image_path}")

#             logger.info("Figure description generated successfully",
#                 extra={
#                     "image_path": str(image_path),
#                     "model": self.model,
#                     "description_length": len(description),
#                 },
#             )

#             return description

#         except VisionModelError:
#             raise

#         except httpx.TimeoutException as exc:
#             logger.exception(
#                 "Vision model request timed out",
#                 extra={
#                     "image_path": str(image_path),
#                     "model": self.model,
#                     "timeout": self.timeout,
#                 },
#             )

#             raise VisionModelError(f"Ollama vision request timed out after {self.timeout} seconds") from exc

#         except httpx.ConnectError as exc:
#             logger.exception("Could not connect to Ollama",
#                 extra={"image_path": str(image_path), "base_url": self.base_url, },
#             )

#             raise VisionModelError(f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running.") from exc

#         except httpx.HTTPStatusError as exc:
#             logger.exception("Ollama returned an HTTP error",
#                 extra={
#                     "image_path": str(image_path),
#                     "model": self.model,
#                     "status_code": exc.response.status_code,
#                 },
#             )

#             raise VisionModelError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}") from exc

#         except httpx.HTTPError as exc:
#             logger.exception("Ollama HTTP request failed",
#                 extra={
#                     "image_path": str(image_path),
#                     "model": self.model,
#                 },
#             )

#             raise VisionModelError(f"Ollama vision request failed: {exc}") from exc

#         except OSError as exc:
#             logger.exception("Could not read figure image",
#                 extra={"image_path": str(image_path), },
#             )

#             raise VisionModelError(f"Could not read figure image {image_path}: {exc}") from exc

#         except Exception as exc:
#             logger.exception("Unexpected error while generating figure description",
#                 extra={
#                     "image_path": str(image_path),
#                     "model": self.model,
#                 },
#             )

#             raise VisionModelError(f"Unexpected vision model error: {exc}") from exc

#     @staticmethod
#     def _build_prompt(caption: str | None) -> str:
#         """
#         Build the prompt used to generate the figure description.
#         """

#         if caption and caption.strip():
#             caption_context = caption.strip()
#         else:
#             caption_context = "No caption available."

#         return f"""
#             You are analyzing a figure from a scientific research paper.

#             Your task is to generate a concise but information-dense description
#             of the figure for a retrieval-augmented generation (RAG) system.

#             The description will be stored in a vector database and retrieved later
#             to answer questions about the research paper.

#             Describe ONLY information that is visually supported by the figure.
#             Do not invent information that is not visible.

#             Focus on the following:

#             1. Figure purpose
#             - What does the figure represent?
#             - What is the main idea or purpose of the figure?

#             2. Components
#             - Identify important components, entities, modules, objects,
#                 stages, or elements visible in the figure.

#             3. Relationships
#             - Explain relationships between components.
#             - Describe arrows, connections, dependencies, or interactions.

#             4. Architecture and workflows
#             - For architecture diagrams, describe the major modules
#                 and how information flows between them.
#             - For pipelines, describe the sequence of stages.
#             - For process diagrams, describe the progression between steps.

#             5. Charts and plots
#             - Identify the chart type when possible.
#             - Describe x-axis and y-axis labels when readable.
#             - Describe legends and series.
#             - Mention important trends, comparisons, peaks, decreases,
#                 increases, or other visible patterns.

#             6. Numerical information
#             - Include visible numerical values only when they are readable.
#             - Do not guess or estimate values that cannot be clearly read.

#             7. Annotations and labels
#             - Include important visible labels, annotations, symbols,
#                 or textual information when readable.

            # 8. Scientific results
            # - Describe important experimental results or comparisons
            #     represented directly in the figure.
            # - Do not infer conclusions that are not supported by the figure.

            # Important rules:

            # - Do not hallucinate missing information.
            # - Do not invent labels, values, relationships, or conclusions.
            # - If text is unreadable, omit it.
            # - Do not say "I can see".
            # - Do not say "the image shows".
            # - Do not discuss image quality.
            # - Do not describe irrelevant visual details such as colors unless
            # the color itself carries scientific meaning.
            # - Use precise scientific terminology when supported by the figure.
            # - Make the description useful for semantic retrieval.

            # Original paper caption:

            # {caption_context}

            # Return ONLY the figure description in plain text.
            # """.strip()