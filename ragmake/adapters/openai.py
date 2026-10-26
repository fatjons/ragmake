from __future__ import annotations

from typing import Any, Sequence

from ragmake.models import Document

_DIGEST = (
    "Summarize the document in 2-4 dense sentences: what it is, its key facts, entities and terms, and what "
    "questions it can answer. Stay grounded in the text. No preamble."
)
_OVERVIEW = (
    "Write a corpus overview from the document digests: scope and purpose, main topics and how documents group, "
    "key entities and terminology, notable gaps or caveats. Mention document ids where useful. Stay grounded. No preamble."
)


class OpenAIEmbedder:
    """Works with openai.OpenAI and openai.AzureOpenAI clients."""

    def __init__(self, client: Any, model: str, *, dimensions: int | None = None, batch_size: int = 128):
        self.client, self.model, self.dimensions, self.batch_size = client, model, dimensions, batch_size
        self.name = f"openai:{model}:{dimensions or 'native'}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=list(texts[i : i + self.batch_size]), **extra)
            out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        return out


class OpenAINectar:
    def __init__(self, client: Any, model: str, *, doc_chars: int = 12000, budget_chars: int = 24000,
                 temperature: float = 0.1, max_tokens: int = 700):
        self.client, self.model, self.doc_chars, self.budget_chars = client, model, doc_chars, budget_chars
        self.temperature, self.max_tokens = temperature, max_tokens

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    def digest(self, document: Document) -> str:
        title = document.title or document.document_id
        return self._chat(_DIGEST, f"Title: {title}\n\n{document.text[: self.doc_chars]}")

    def reduce(self, corpus_id: str, digests: Sequence[tuple[str, str]], previous: str | None) -> str:
        lines = [f"- {doc_id}: {digest}" for doc_id, digest in digests]
        if sum(map(len, lines)) > self.budget_chars:
            batches: list[list[str]] = [[]]
            size = 0
            for line in lines:
                if batches[-1] and size + len(line) > self.budget_chars:
                    batches.append([])
                    size = 0
                batches[-1].append(line)
                size += len(line)
            partials = [self._chat(_OVERVIEW, "\n".join(b)) for b in batches]
            return self.reduce(corpus_id, [(f"part {i}", p) for i, p in enumerate(partials, 1)], previous)
        user = f"Corpus: {corpus_id}\n\n" + "\n".join(lines)
        if previous:
            user += f"\n\nPrevious version, for continuity; prefer current digests:\n{previous}"
        return self._chat(_OVERVIEW, user)
