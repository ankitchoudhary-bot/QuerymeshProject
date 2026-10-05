"""Agent discovery registry.

Lets the supervisor (and, later, A2A/MCP callers) discover agents by
capability rather than hardcoding which concrete agent class handles
which job. Each agent is registered with an `AgentCard` describing what
it does; the registry never inspects agent internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_KNOWN_CAPABILITIES = {"research", "retrieval", "grading", "synthesis", "citation"}


class AgentDiscoveryError(Exception):
    """Base error for agent registry failures."""


class AgentAlreadyRegisteredError(AgentDiscoveryError):
    """Raised when registering a name that's already taken."""


class AgentNotFoundError(AgentDiscoveryError):
    """Raised when looking up a name that isn't registered."""


@dataclass(frozen=True)
class AgentCard:
    """Metadata an agent advertises to the registry."""

    name: str
    capabilities: list[str] = field(default_factory=list)
    description: str = ""


class AgentRegistry:
    """In-memory registry mapping agent names/capabilities to instances."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[AgentCard, Any]] = {}

    def register(self, card: AgentCard, instance: Any) -> None:
        if card.name in self._entries:
            raise AgentAlreadyRegisteredError(f"Agent '{card.name}' is already registered")
        self._entries[card.name] = (card, instance)

    def get(self, name: str) -> Any:
        try:
            return self._entries[name][1]
        except KeyError as exc:
            raise AgentNotFoundError(f"No agent registered as '{name}'") from exc

    def discover(self, capability: str) -> list[Any]:
        """Return all agent instances advertising the given capability."""
        return [
            instance
            for card, instance in self._entries.values()
            if capability in card.capabilities
        ]

    def list_agents(self) -> list[AgentCard]:
        return [card for card, _ in self._entries.values()]
