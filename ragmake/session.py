from __future__ import annotations

from .compiler import _q, _utcnow
from .interfaces import ArtifactStore


class SessionStore:
    def __init__(self, artifacts: ArtifactStore, namespace: str = "ragmake", max_events: int = 200):
        self.artifacts, self.ns, self.max_events = artifacts, namespace, max_events

    def _path(self, session_id: str) -> str:
        return f"{self.ns}/sessions/{_q(session_id)}.json"

    def load(self, session_id: str) -> dict | None:
        return self.artifacts.read_json(self._path(session_id))

    def record(self, session_id: str, kind: str, text: str, *, concepts=(), entities=(), **meta) -> dict:
        now = _utcnow()
        state = self.load(session_id) or {"session_id": session_id, "created_at": now,
                                          "concepts": [], "entities": [], "events": []}
        state["updated_at"] = now
        state["concepts"] = sorted({*state["concepts"], *concepts})
        state["entities"] = sorted({*state["entities"], *entities})
        state["events"] = [*state["events"], {"kind": kind, "text": text, "at": now, **meta}][-self.max_events:]
        self.artifacts.write_json(self._path(session_id), state)
        return state

    def summary(self, session_id: str, recent: int = 3) -> dict | None:
        state = self.load(session_id)
        if state is None:
            return None
        by_kind: dict[str, list[str]] = {}
        for event in state["events"]:
            by_kind.setdefault(event["kind"], []).append(event["text"])
        return {"concepts": state["concepts"], "entities": state["entities"],
                "recent": {kind: texts[-recent:] for kind, texts in by_kind.items()}}
