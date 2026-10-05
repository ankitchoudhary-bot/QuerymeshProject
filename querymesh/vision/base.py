from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionModel(Protocol):
    async def describe(self, image_path: Path, caption: str | None = None, ) -> str:
        ...