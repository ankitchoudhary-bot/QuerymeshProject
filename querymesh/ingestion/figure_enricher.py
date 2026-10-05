from __future__ import annotations
from pathlib import Path
from querymesh.ingestion.document_models import ParsedDocument
from querymesh.vision.base import VisionModel
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

class FigureEnricher:
    """Generate semantic descriptions for extracted figures."""

    def __init__(self, vision_model: VisionModel, ) -> None:
        self.vision_model = vision_model

    async def enrich(self, document: ParsedDocument, ) -> ParsedDocument:

        for index, figure in enumerate(document.figures, start=1, ):
            if not figure.image_path:
                logger.warning("Figure %s has no image path", index, )
                continue

            try:
                description = (await self.vision_model.describe(image_path=Path(figure.image_path), caption=figure.caption, ))

                figure.description = description

                logger.info(
                    "Generated description for figure %s "
                    "of %s",
                    index,
                    document.arxiv_id,
                )

            except Exception as exc:
                logger.warning("Failed to describe figure %s: %s", index, exc, )

        return document
