"""Persistent cosine index and reciprocal rank fusion for the two SOMA collections."""
from functools import lru_cache
from hashlib import sha256
import logging
import os
from pathlib import Path
from time import sleep

from django.conf import settings
from google.genai.errors import APIError
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np


logger = logging.getLogger(__name__)
DIMENSIONS = 256
DATA = Path(__file__).resolve().parent.parent / "data"


def embeddings_configured():
    return bool(settings.GOOGLE_API_KEY)


@lru_cache(maxsize=2)
def embedding_model(timeout):
    return GoogleGenerativeAIEmbeddings(
        model=settings.GOOGLE_EMBED_MODEL,
        output_dimensionality=DIMENSIONS,
        api_key=settings.GOOGLE_API_KEY,
        request_options={"timeout": timeout},
    )


def embed(texts, timeout, query=False):
    try:
        model = embedding_model(timeout)
        result = [model.embed_query(texts[0])] if query else model.embed_documents(texts)
    except APIError as exc:
        raise ValueError(f"No se pudieron generar embeddings con Gemini: {exc}") from exc
    values = np.asarray(result, dtype=np.float32)
    if values.shape != (len(texts), DIMENSIONS) or not np.isfinite(values).all():
        raise ValueError("Dimensiones de embeddings inesperadas.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("Gemini devolvió embeddings vacíos.")
    return values / norms


def texts_for(docs):
    return [f'{doc.metadata["title"]}\n{doc.metadata["excerpt"]}' for doc in docs]


def fingerprint(texts):
    digest = sha256()
    for text in texts:
        part = text.encode()
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def save_index(path, key, vectors):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, fingerprint=np.asarray(key),
                            model=np.asarray(settings.GOOGLE_EMBED_MODEL), vectors=vectors.astype(np.float16))
    tmp.replace(path)


def read_index(path, key, model):
    with np.load(path, allow_pickle=False) as archive:
        if archive["fingerprint"].item() != key or archive["model"].item() != model:
            raise ValueError("El índice corresponde a otro corpus o modelo.")
        vectors = archive["vectors"].astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSIONS or not np.isfinite(vectors).all():
        raise ValueError("Índice vectorial incompatible.")
    return vectors


def build_vectors(collection, docs, progress):
    texts = texts_for(docs)
    key = fingerprint(texts)
    path = DATA / f"{collection}-vectors.npz"
    checkpoint = DATA / f"{collection}-vectors.partial.npz"
    try:
        current = read_index(path, key, settings.GOOGLE_EMBED_MODEL)
        if len(current) == len(texts):
            return path
    except (OSError, ValueError, KeyError):
        pass
    try:
        vectors = read_index(checkpoint, key, settings.GOOGLE_EMBED_MODEL)
        if len(vectors) > len(texts):
            raise ValueError("Checkpoint incompatible.")
    except (OSError, ValueError, KeyError):
        vectors = np.empty((0, DIMENSIONS), dtype=np.float32)
    for start in range(len(vectors), len(texts), 32):
        batch = texts[start:start + 32]
        for attempt in range(3):
            try:
                embeddings = embed(batch, timeout=160)
                break
            except (OSError, ValueError, KeyError):
                if attempt == 2:
                    raise
                sleep(2 ** attempt)
        vectors = np.concatenate((vectors, embeddings))
        save_index(checkpoint, key, vectors)
        progress(f"{collection}: {len(vectors)}/{len(texts)}", ending="\n")
    os.replace(checkpoint, path)
    return path


@lru_cache(maxsize=6)
def cached_vectors(path, _mtime, data_hash, model):
    return read_index(path, data_hash, model)


def hybrid_rank(collection, docs, question, lexical, eligible=None):
    if not embeddings_configured():
        return lexical
    path = DATA / f"{collection}-vectors.npz"
    if not path.is_file():
        return lexical
    try:
        key = fingerprint(texts_for(docs))
        vectors = cached_vectors(path, path.stat().st_mtime_ns, key, settings.GOOGLE_EMBED_MODEL)
        if len(vectors) != len(docs):
            return lexical
        query = embed([question], timeout=12, query=True)[0]
        similarities = vectors @ query
    except (OSError, ValueError, KeyError):
        logger.warning("Búsqueda vectorial no disponible para %s. Se usa BM25.", collection)
        return lexical

    semantic = [int(i) for i in np.argsort(-similarities, kind="stable")
                if similarities[i] >= settings.RAG_EMBED_MIN_SCORE and (eligible is None or int(i) in eligible)][:30]
    scores = {}
    for ranking in (lexical[:30], semantic):
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0) + 1 / (40 + rank)
    return sorted(scores, key=scores.get, reverse=True)
