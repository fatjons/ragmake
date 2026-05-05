from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .interfaces import ArtifactStore


@dataclass(slots=True)
class LearningEvent:
    timestamp: str
    summary: str
    concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IntentEvent:
    timestamp: str
    intent: str
    metadata: dict[str, Any] = field(default_factory=dict)
    influenced_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionEvent:
    timestamp: str
    decision: str
    rationale: str | None = None
    influenced_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at: str
    updated_at: str
    learned_concepts: list[str] = field(default_factory=list)
    discussed_entities: list[str] = field(default_factory=list)
    learning_events: list[LearningEvent] = field(default_factory=list)
    intents: list[IntentEvent] = field(default_factory=list)
    decisions: list[DecisionEvent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            learned_concepts=list(data.get("learned_concepts") or []),
            discussed_entities=list(data.get("discussed_entities") or []),
            learning_events=[
                LearningEvent(
                    timestamp=e["timestamp"],
                    summary=e["summary"],
                    concepts=list(e.get("concepts") or []),
                    entities=list(e.get("entities") or []),
                )
                for e in data.get("learning_events") or []
            ],
            intents=[
                IntentEvent(
                    timestamp=e["timestamp"],
                    intent=e["intent"],
                    metadata=dict(e.get("metadata") or {}),
                    influenced_by=list(e.get("influenced_by") or []),
                )
                for e in data.get("intents") or []
            ],
            decisions=[
                DecisionEvent(
                    timestamp=e["timestamp"],
                    decision=e["decision"],
                    rationale=e.get("rationale"),
                    influenced_by=list(e.get("influenced_by") or []),
                )
                for e in data.get("decisions") or []
            ],
        )


class SessionStateManager:
    def __init__(self, artifact_store: ArtifactStore, namespace: str = ".stateful_rag"):
        self.artifact_store = artifact_store
        self.namespace = namespace.rstrip("/")

    def load(self, session_id: str) -> SessionState | None:
        data = self.artifact_store.read_json(self._session_path(session_id))
        return SessionState.from_dict(data) if data else None

    def record_learning(
        self,
        session_id: str,
        summary: str,
        concepts: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> SessionState:
        state = self._ensure_state(session_id)
        event = LearningEvent(
            timestamp=_utcnow(),
            summary=summary,
            concepts=sorted(set(concepts or [])),
            entities=sorted(set(entities or [])),
        )
        state.learning_events.append(event)
        state.learned_concepts = sorted(set(state.learned_concepts).union(event.concepts))
        state.discussed_entities = sorted(set(state.discussed_entities).union(event.entities))
        return self._save(state)

    def record_intent(
        self,
        session_id: str,
        intent: str,
        metadata: dict[str, Any] | None = None,
        influenced_by: list[str] | None = None,
    ) -> SessionState:
        state = self._ensure_state(session_id)
        state.intents.append(
            IntentEvent(
                timestamp=_utcnow(),
                intent=intent,
                metadata=dict(metadata or {}),
                influenced_by=sorted(set(influenced_by or [])),
            )
        )
        return self._save(state)

    def record_decision(
        self,
        session_id: str,
        decision: str,
        rationale: str | None = None,
        influenced_by: list[str] | None = None,
    ) -> SessionState:
        state = self._ensure_state(session_id)
        state.decisions.append(
            DecisionEvent(
                timestamp=_utcnow(),
                decision=decision,
                rationale=rationale,
                influenced_by=sorted(set(influenced_by or [])),
            )
        )
        return self._save(state)

    def _ensure_state(self, session_id: str) -> SessionState:
        existing = self.load(session_id)
        if existing:
            return existing
        now = _utcnow()
        return SessionState(session_id=session_id, created_at=now, updated_at=now)

    def _save(self, state: SessionState) -> SessionState:
        state.updated_at = _utcnow()
        self.artifact_store.write_json(self._session_path(state.session_id), asdict(state))
        return state

    def _session_path(self, session_id: str) -> str:
        return f"{self.namespace}/sessions/{session_id}.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
