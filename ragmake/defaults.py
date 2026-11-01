from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Sequence

from .models import Document
from .storage import normalize

_STOPWORDS = frozenset(
    "a about above after again all also am an and any are as at be because been before being below between both but "
    "by can could did do does doing down during each few for from further had has have having he her here hers him his "
    "how i if in into is it its just me more most my no nor not now of off on once only or other our out over own same "
    "she should so some such than that the their them then there these they this those through to too under until up "
    "very was we were what when where which while who whom why will with would you your".split()
)
_WORD = re.compile(r"[a-z0-9_]{2,}")


class HashingEmbedder:
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}"

    def embed_texts(self, texts: Sequence[str], query: bool = False) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for word in _WORD.findall(text.lower()):
                h = hashlib.sha256(word.encode()).digest()
                vec[int.from_bytes(h[:4], "big") % self.dimensions] += 1.0 if h[4] & 1 else -1.0
            out.append(normalize(vec))
        return out


class HeuristicNectar:
    def __init__(self, max_terms: int = 10, lead_chars: int = 240, budget_chars: int = 8000):
        self.max_terms, self.lead_chars, self.budget_chars = max_terms, lead_chars, budget_chars
        self.name = f"heuristic-{max_terms}-{lead_chars}-{budget_chars}"

    def digest(self, document: Document) -> str:
        text = " ".join(document.text.split())
        lead = text[: self.lead_chars]
        if len(text) > self.lead_chars and (i := lead.rfind(". ")) > 0:
            lead = lead[:i]
        words = [w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS]
        terms = ", ".join(w for w, _ in Counter(words).most_common(self.max_terms))
        title = f"{document.title}: " if document.title else ""  # the id is already beside the digest
        return f"{title}{lead.rstrip('. ')}. Key terms: {terms}"

    def reduce(self, corpus_id: str, digests: Sequence[tuple[str, str]], previous: str | None) -> str:
        terms = [d.rpartition("Key terms: ")[2].split(", ") for _, d in digests]
        recurring = Counter(t for doc_terms in terms for t in dict.fromkeys(doc_terms) if t)  # ordered: ties are stable
        topics: dict[str, list[str]] = {}
        for (doc_id, _), doc_terms in zip(digests, terms):
            if doc_terms[0]:
                topics.setdefault(doc_terms[0], []).append(doc_id)
        head = [f"Corpus {corpus_id}: {len(digests)} documents.",
                "Recurring terms: " + ", ".join(t for t, _ in recurring.most_common(self.max_terms))]
        budget = self.budget_chars - sum(len(line) + 1 for line in head) - len("Topics:\n\nDocuments:")
        shared = sorted((item for item in topics.items() if len(item[1]) > 1), key=lambda kv: -len(kv[1]))
        topic_lines = _fit([f"- {t} ({len(ids)} docs): {', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''}"
                            for t, ids in shared], budget // 2, "topics")
        budget -= sum(len(line) + 1 for line in topic_lines)
        doc_lines = _fit([f"- {doc_id}: {digest}" for doc_id, digest in digests], budget, "documents")
        topics_section = ["Topics:", *topic_lines] if topic_lines else []
        return "\n".join(head + topics_section + ["Documents:"] + doc_lines)


def _fit(lines: list[str], budget: int, noun: str) -> list[str]:
    """The lines that fit in budget chars, ending with '- ... and k more' when they don't all fit."""
    if sum(len(line) + 1 for line in lines) <= budget:
        return lines
    out, size = [], len(f"- ... and {len(lines)} more {noun}") + 1  # longest possible tail, reserved
    for line in lines:
        if size + len(line) + 1 > budget:
            break
        out.append(line)
        size += len(line) + 1
    return out + [f"- ... and {len(lines) - len(out)} more {noun}"]
