from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


load_dotenv(override=True)


class MTRetriever:
    def __init__(
        self,
        db_name: str,
        encoder: str,
        source: str,
        target: str,
        retrieval_backend: str = "faiss",
        bm25_index_path: str | None = None,
        ensemble_faiss_weight: float = 0.5,
        ensemble_bm25_weight: float = 0.5,
    ) -> None:
        self.db_path = db_name
        self.encoder = encoder
        self.source = source
        self.target = target
        self.source_set = set()
        self.db = None

        self.retrieval_backend = retrieval_backend.lower().strip()
        if self.retrieval_backend not in {"faiss", "bm25", "ensemble"}:
            raise ValueError(
                "retrieval_backend must be one of: faiss, bm25, ensemble"
            )

        self.ensemble_faiss_weight = float(ensemble_faiss_weight)
        self.ensemble_bm25_weight = float(ensemble_bm25_weight)

        self.bm25_index_path = (
            str(bm25_index_path)
            if bm25_index_path
            else f"{self.db_path}_bm25.json"
        )
        self.bm25_docs: list[dict[str, Any]] = []
        self.bm25_model = None

        print(
            "[retriever] "
            f"source:{source} target:{target} backend:{self.retrieval_backend}"
        )

        self.embedding_model = None
        if self.retrieval_backend in {"faiss", "ensemble"}:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.encoder)
            if os.path.exists(self.db_path):
                print(f"Loading existing FAISS DB from {self.db_path}")
                self.db = FAISS.load_local(
                    self.db_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )

        if self.retrieval_backend in {"bm25", "ensemble"}:
            if BM25Okapi is None:
                raise ImportError(
                    "rank-bm25 is required for BM25 retrieval. "
                    "Install with: pip install rank-bm25"
                )
            self._load_bm25_index()

        if self.db is None and not self.bm25_docs:
            raise FileNotFoundError(
                "Few-shot retrieval index was not found. "
                f"Expected FAISS directory '{self.db_path}' or BM25 file "
                f"'{self.bm25_index_path}'. Set MFDS_FAISS_DB_ROOT to the prepared "
                "index prefix described in README.md."
            )

        # If BM25 index is missing but FAISS is available, bootstrap BM25 from FAISS docs.
        if self.retrieval_backend in {"bm25", "ensemble"} and not self.bm25_docs:
            self._bootstrap_bm25_from_faiss()

    @staticmethod
    def _extract_doc_id(metadata: dict[str, Any] | None) -> str | None:
        metadata = metadata or {}
        doc_id = (
            metadata.get("doc_id")
            or metadata.get("document_id")
            or metadata.get("source_doc_id")
        )
        if isinstance(doc_id, str) and doc_id:
            return doc_id
        return None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [tok for tok in text.lower().split() if tok]

    def _refresh_bm25_model(self) -> None:
        if BM25Okapi is None:
            self.bm25_model = None
            return
        tokenized = [self._tokenize(doc["source"]) for doc in self.bm25_docs]
        self.bm25_model = BM25Okapi(tokenized) if tokenized else None

    def _save_bm25_index(self) -> None:
        index_path = Path(self.bm25_index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "source_lang": self.source,
            "target_lang": self.target,
            "documents": self.bm25_docs,
        }
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print(f"[retriever] saved BM25 index: {index_path}")

    def _load_bm25_index(self) -> None:
        index_path = Path(self.bm25_index_path)
        if not index_path.exists():
            self.bm25_docs = []
            self.bm25_model = None
            return

        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        docs = payload.get("documents", [])
        cleaned: list[dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            src = doc.get("source")
            ref = doc.get("reference")
            if not isinstance(src, str) or not isinstance(ref, str):
                continue
            src = src.strip()
            ref = ref.strip()
            if not src or not ref:
                continue
            row = {"source": src, "reference": ref}
            doc_id = doc.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                row["doc_id"] = doc_id
            cleaned.append(row)

        self.bm25_docs = cleaned
        self._refresh_bm25_model()
        print(f"[retriever] loaded BM25 docs: {len(self.bm25_docs)}")

    def _faiss_docs(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        if self.db is None:
            return docs

        index_map = getattr(self.db, "index_to_docstore_id", {})
        docstore = getattr(self.db, "docstore", None)
        for idx in sorted(index_map):
            doc_key = index_map[idx]
            doc = docstore.search(doc_key) if docstore is not None else None
            if doc is None or not hasattr(doc, "page_content"):
                continue
            src = doc.page_content
            metadata = getattr(doc, "metadata", {}) or {}
            docs.append(
                {
                    "source": src,
                    "reference": metadata.get("reference", None),
                    "doc_id": self._extract_doc_id(metadata),
                }
            )
        return docs

    def _bootstrap_bm25_from_faiss(self) -> None:
        faiss_docs = self._faiss_docs()
        if not faiss_docs:
            return
        self.bm25_docs = [
            {
                "source": row["source"],
                "reference": row.get("reference"),
                "doc_id": row.get("doc_id"),
            }
            for row in faiss_docs
            if isinstance(row.get("source"), str) and row["source"]
        ]
        self._refresh_bm25_model()
        self._save_bm25_index()

    def add(self, fewshots: List[Any]):
        print("[retriever] Adding entries")

        if len(self.source_set) == 0:
            if self.db is not None:
                for doc in self._faiss_docs():
                    self.source_set.add(doc["source"])
            elif self.bm25_docs:
                for doc in self.bm25_docs:
                    self.source_set.add(doc["source"])

        texts = []
        metadatas = []
        bm25_new_docs: list[dict[str, Any]] = []
        for item in fewshots:
            src = item["source"]
            ref = item["reference"]
            if src in self.source_set:
                continue

            metadata = {
                "reference": ref,
                "source": src,
            }
            doc_id = item.get("doc_id")
            if doc_id:
                metadata["doc_id"] = doc_id

            texts.append(src)
            metadatas.append(metadata)
            bm25_doc = {"source": src, "reference": ref}
            if doc_id:
                bm25_doc["doc_id"] = doc_id
            bm25_new_docs.append(bm25_doc)

            self.source_set.add(src)

        if not texts:
            print("[retriever] no text, ending add")
            return

        if self.retrieval_backend in {"faiss", "ensemble"}:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model is not initialized for FAISS backend")
            if self.db is None:
                print("[retriever] creating new faiss")
                os.makedirs(self.db_path, exist_ok=True)
                self.db = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
                print("[retriever] FAISS creation done")
            else:
                self.db.add_texts(texts, metadatas=metadatas)
            self.db.save_local(self.db_path)

        if self.retrieval_backend in {"bm25", "ensemble"}:
            self.bm25_docs.extend(bm25_new_docs)
            self._refresh_bm25_model()
            self._save_bm25_index()

    def _search_faiss(
        self,
        sent: str,
        top_k: int,
        mode: str,
        exclude_sources: set[str],
        exclude_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        if self.db is None:
            return []

        doc_count = len(getattr(self.db, "index_to_docstore_id", {}))
        if doc_count == 0:
            return []

        base_k = top_k + 1 if mode == "train" else top_k
        search_k = min(doc_count, max(base_k + 20, base_k * 5))
        results = self.db.similarity_search_with_score(sent, k=search_k)

        formatted = []
        removed_train_self = False
        for result in results:
            doc = result[0]
            score = float(result[1])
            metadata = doc.metadata or {}
            src = doc.page_content
            doc_id = self._extract_doc_id(metadata)

            if mode == "train" and not removed_train_self and src == sent:
                removed_train_self = True
                continue
            if src in exclude_sources:
                continue
            if doc_id in exclude_doc_ids:
                continue

            formatted.append(
                {
                    "src": src,
                    "mt": metadata.get("reference", None),
                    "doc_id": doc_id,
                    "score": score,
                }
            )
            if len(formatted) >= top_k:
                break
        return formatted

    def _search_bm25(
        self,
        sent: str,
        top_k: int,
        mode: str,
        exclude_sources: set[str],
        exclude_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not self.bm25_docs or self.bm25_model is None:
            return []

        query_tokens = self._tokenize(sent)
        scores = self.bm25_model.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        formatted = []
        removed_train_self = False
        for idx in ranked_idx:
            row = self.bm25_docs[idx]
            src = row["source"]
            doc_id = row.get("doc_id")

            if mode == "train" and not removed_train_self and src == sent:
                removed_train_self = True
                continue
            if src in exclude_sources:
                continue
            if doc_id in exclude_doc_ids:
                continue

            formatted.append(
                {
                    "src": src,
                    "mt": row.get("reference", None),
                    "doc_id": doc_id,
                    "score": float(scores[idx]),
                }
            )
            if len(formatted) >= top_k:
                break
        return formatted

    def _search_ensemble(
        self,
        sent: str,
        top_k: int,
        mode: str,
        exclude_sources: set[str],
        exclude_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 5, top_k + 20)
        faiss_hits = self._search_faiss(
            sent,
            top_k=candidate_k,
            mode=mode,
            exclude_sources=exclude_sources,
            exclude_doc_ids=exclude_doc_ids,
        )
        bm25_hits = self._search_bm25(
            sent,
            top_k=candidate_k,
            mode=mode,
            exclude_sources=exclude_sources,
            exclude_doc_ids=exclude_doc_ids,
        )

        if not faiss_hits:
            return bm25_hits[:top_k]
        if not bm25_hits:
            return faiss_hits[:top_k]

        fused: dict[str, dict[str, Any]] = {}
        rrf_base = 60.0

        for rank, hit in enumerate(faiss_hits, start=1):
            src = hit["src"]
            record = fused.setdefault(
                src,
                {
                    "src": src,
                    "mt": hit.get("mt"),
                    "doc_id": hit.get("doc_id"),
                    "score": 0.0,
                },
            )
            record["score"] += self.ensemble_faiss_weight / (rrf_base + rank)

        for rank, hit in enumerate(bm25_hits, start=1):
            src = hit["src"]
            record = fused.setdefault(
                src,
                {
                    "src": src,
                    "mt": hit.get("mt"),
                    "doc_id": hit.get("doc_id"),
                    "score": 0.0,
                },
            )
            if record.get("mt") is None:
                record["mt"] = hit.get("mt")
            if record.get("doc_id") is None:
                record["doc_id"] = hit.get("doc_id")
            record["score"] += self.ensemble_bm25_weight / (rrf_base + rank)

        ranked = sorted(fused.values(), key=lambda row: row["score"], reverse=True)
        return ranked[:top_k]

    def _fixed_candidates(
        self,
        exclude_sources: set[str],
        exclude_doc_ids: set[str],
    ) -> list[dict[str, Any]]:
        if self.retrieval_backend in {"bm25", "ensemble"} and self.bm25_docs:
            candidates = []
            for row in self.bm25_docs:
                src = row["source"]
                doc_id = row.get("doc_id")
                if src in exclude_sources:
                    continue
                if doc_id in exclude_doc_ids:
                    continue
                candidates.append(
                    {
                        "src": src,
                        "mt": row.get("reference", None),
                        "score": 0.0,
                    }
                )
            return candidates

        candidates = []
        for row in self._faiss_docs():
            src = row["source"]
            doc_id = row.get("doc_id")
            if src in exclude_sources:
                continue
            if doc_id in exclude_doc_ids:
                continue
            candidates.append(
                {
                    "src": src,
                    "mt": row.get("reference", None),
                    "score": 0.0,
                }
            )
        return candidates

    def search(
        self,
        sentences: List[str],
        top_k: int,
        mode: str = "test",
        exclude_sources=None,
        exclude_doc_ids=None,
        use_fixed: bool = False,
        fixed_seed: int = 42,
    ):
        if self.db is None and not self.bm25_docs:
            return []

        if isinstance(sentences, str):
            sentences = [sentences]

        exclude_sources = {
            src for src in (exclude_sources or set()) if isinstance(src, str) and src
        }
        exclude_doc_ids = {
            doc_id
            for doc_id in (exclude_doc_ids or set())
            if isinstance(doc_id, str) and doc_id
        }

        if use_fixed:
            candidates = self._fixed_candidates(exclude_sources, exclude_doc_ids)
            if len(candidates) > top_k:
                rng = random.Random(fixed_seed)
                sampled_indices = rng.sample(range(len(candidates)), k=top_k)
                sampled_indices.sort()
                selected = [candidates[i] for i in sampled_indices]
            else:
                selected = candidates
            return [[dict(item) for item in selected] for _ in sentences]

        fewshots = []
        for sent in sentences:
            if self.retrieval_backend == "faiss":
                formatted = self._search_faiss(
                    sent,
                    top_k=top_k,
                    mode=mode,
                    exclude_sources=exclude_sources,
                    exclude_doc_ids=exclude_doc_ids,
                )
            elif self.retrieval_backend == "bm25":
                formatted = self._search_bm25(
                    sent,
                    top_k=top_k,
                    mode=mode,
                    exclude_sources=exclude_sources,
                    exclude_doc_ids=exclude_doc_ids,
                )
            else:
                formatted = self._search_ensemble(
                    sent,
                    top_k=top_k,
                    mode=mode,
                    exclude_sources=exclude_sources,
                    exclude_doc_ids=exclude_doc_ids,
                )
            fewshots.append(
                [
                    {
                        "src": row["src"],
                        "mt": row.get("mt"),
                        "score": float(row.get("score", 0.0)),
                    }
                    for row in formatted
                ]
            )
        return fewshots

    def delete(self, sentences: List[str]):
        remove_set = set(sentences)

        if self.retrieval_backend in {"faiss", "ensemble"}:
            if self.db is not None:
                docs = self._faiss_docs()
                keep_texts = []
                keep_metadatas = []
                for row in docs:
                    if row["source"] in remove_set:
                        continue
                    keep_texts.append(row["source"])
                    metadata = {"reference": row.get("reference")}
                    if row.get("doc_id"):
                        metadata["doc_id"] = row["doc_id"]
                    keep_metadatas.append(metadata)

                if keep_texts:
                    if self.embedding_model is None:
                        raise RuntimeError("Embedding model is not initialized")
                    self.db = FAISS.from_texts(
                        keep_texts,
                        self.embedding_model,
                        metadatas=keep_metadatas,
                    )
                    self.db.save_local(self.db_path)
                else:
                    self.db = None

        if self.retrieval_backend in {"bm25", "ensemble"}:
            self.bm25_docs = [
                row for row in self.bm25_docs if row.get("source") not in remove_set
            ]
            self._refresh_bm25_model()
            self._save_bm25_index()

        self.source_set = set()
        if self.retrieval_backend in {"bm25", "ensemble"}:
            self.source_set.update(row["source"] for row in self.bm25_docs)
        elif self.db is not None:
            self.source_set.update(row["source"] for row in self._faiss_docs())

if __name__ == "__main__":
    rt = MTRetriever(
        db_name="split_data_train_en_to_kr",
        encoder="BAAI/bge-m3",
        source="en",
        target="kr",
        retrieval_backend="ensemble",
    )
    rt.source_set = set()

    fewshots = [
        {"source": "hello world", "reference": "ref1"},
        {"source": "bonjour", "reference": "ref2"},
    ]
    rt.add(fewshots)
    print("After add:", rt.search(["test query"], top_k=5))

    rt.delete(["hello world"])
    print("After delete:", rt.search(["test query"], top_k=5))
