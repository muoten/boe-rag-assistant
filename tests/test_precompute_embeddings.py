"""Tests for the embedding precompute script and its corpus check, without a model."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from boe_rag.eval.retriever import FloatMatrix, load_embeddings, save_embeddings

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _script(name: str) -> ModuleType:
    """Import a script by file name."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_corpus(path: Path) -> None:
    table = pa.table(
        {
            "chunk_id": ["a", "b"],
            "document_title": ["Ley 1/2020", "Ley 2/2020"],
            "text": ["uno", "dos"],
        }
    )
    pq.write_table(table, path)


@pytest.mark.parametrize(
    ("with_titles", "expected"),
    [
        (False, ["uno", "dos"]),
        (True, ["Ley 1/2020. uno", "Ley 2/2020. dos"]),
    ],
)
def test_precompute_embeds_titles_only_when_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_titles: bool,
    expected: list[str],
) -> None:
    """--with-titles prefixes each passage with its title; the ids stay in order."""
    seen: list[str] = []

    class _Embedder:
        def __init__(self, model_name: str) -> None:
            pass

        def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
            seen.extend(texts)
            return np.zeros((len(texts), 3), dtype=np.float32)

    module = _script("precompute_embeddings")
    monkeypatch.setattr(module, "E5Embedder", _Embedder)
    corpus, out = tmp_path / "c.parquet", tmp_path / "e.npz"
    _write_corpus(corpus)

    module.precompute(corpus, out, "model", with_titles=with_titles)

    assert seen == expected
    assert load_embeddings(out)[0] == ["a", "b"]


@pytest.mark.parametrize(
    ("ids", "code"), [(["a", "b"], 0), (["b", "a"], 1), (["a"], 1)]
)
def test_check_embeddings_needs_the_same_ids_in_order(
    tmp_path: Path, ids: list[str], code: int
) -> None:
    """A matrix passes only if its ids match the corpus, order included."""
    corpus, embeddings = tmp_path / "c.parquet", tmp_path / "e.npz"
    _write_corpus(corpus)
    save_embeddings(embeddings, ids, np.zeros((len(ids), 3), dtype=np.float32))

    argv = ["--corpus", str(corpus), "--embeddings", str(embeddings)]
    assert _script("check_embeddings").main(argv) == code
