"""Pre-download the serving models into the local Hugging Face cache.

Run at Docker build time so the embedding and cross-encoder weights are baked
into the image. The running Space then loads them from cache instead of
downloading on first request, removing that from the cold-start path.

Requires the ``ml`` extra (``pip install -e .[ml]``).

``--download-only`` fetches the same weights without importing PyTorch or loading
the models — for cross-platform builds (an amd64 image built on an ARM machine),
where running the models under emulation can take far longer than the download.

Example:
    python scripts/warm_models.py
    python scripts/warm_models.py --download-only
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


#: Files a sentence-transformers load never reads; skipped by ``--download-only``.
_UNUSED_WEIGHTS = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"]


def download_models() -> None:
    """Fetch each serving model's files into the cache without loading it."""
    from huggingface_hub import snapshot_download

    from boe_rag.eval.cross_encoder import DEFAULT_RERANK_MODEL
    from boe_rag.eval.embedding import DEFAULT_MODEL

    for model in (DEFAULT_MODEL, DEFAULT_RERANK_MODEL):
        logger.info("Downloading %s ...", model)
        snapshot_download(model, ignore_patterns=_UNUSED_WEIGHTS)
    logger.info("Models cached.")


def warm_models() -> None:
    """Instantiate each serving model so its weights are fetched and cached."""
    from boe_rag.eval.cross_encoder import CrossEncoderReranker
    from boe_rag.eval.embedding import E5Embedder

    logger.info("Warming the embedding model ...")
    E5Embedder()
    logger.info("Warming the cross-encoder model ...")
    CrossEncoderReranker()
    logger.info("Models cached.")


def main() -> int:
    """Download and cache the serving models.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="fetch the weights without loading the models",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.download_only:
        download_models()
    else:
        warm_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
