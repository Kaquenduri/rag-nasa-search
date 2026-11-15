from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
STOPWORDS = set("""
a al algo ambos como cual cuales cuando cuanto de del donde el en entre era es esa ese
esta este fueron fue ha hay la las le les lo los me para por que quien se si sin sobre
son su sus te un una unas unos y o ahora hoy actual actualidad the is on in which what did of was for and a used
usa usaban usaron uso sirven sirvio llevo llevaron llevaba llevaban diferencia
comparar compara dame dime explica tiene tienen podria suelen generalmente
""".split())
ALIASES = {
    "apolo": "apollo", "luna": "moon", "lunar": "moon", "alunizaje": "landing",
    "alunizo": "landing", "aterrizaje": "landing", "aterrizo": "landing",
    "misiones": "mission", "mision": "mission", "nave": "spacecraft",
    "modulo": "module", "motor": "engine", "antimateria": "antimatter", "mando": "command", "superficie": "surface",
    "tripulacion": "crew", "retorno": "return", "equipo": "equipment", "equipos": "equipment",
    "instrumentos": "instrument", "instrumento": "instrument", "instruments": "instrument",
    "caminatas": "spacewalk", "caminata": "spacewalk", "spacewalks": "spacewalk",
    "traje": "spacesuit", "trajes": "spacesuit", "spacesuits": "spacesuit",
    "herramientas": "tool", "herramienta": "tool", "tools": "tool",
    "muestras": "sample", "muestra": "sample", "samples": "sample",
    "recoleccion": "collection", "recoger": "collect", "recogieron": "collect",
    "bolsa": "bag", "bolsas": "bag", "bags": "bag", "taladro": "drill",
    "martillo": "hammer", "pala": "scoop", "pinzas": "tongs",
    "altimetro": "altimeter", "espectrometro": "spectrometer", "rayos": "ray",
    "sismico": "seismic", "sismicos": "seismic", "magnetometro": "magnetometer",
    "camara": "camera", "camaras": "camera", "fotografia": "photography",
    "vehiculo": "vehicle", "roving": "rover", "excursion": "eva",
    "comunicaciones": "communications", "saturno": "saturn", "jupiter": "jupiter",
    "estacion": "station", "espacial": "space",
    "orbitales": "orbital", "orbita": "orbit", "sonda": "probe",
}
NUMBERED = re.compile(r"\b(apollo|apolo|voyager|viking|mariner|surveyor|gemini|mercury)\s*[- ]?\s*(\d{1,2})\b")


def normal(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text):
    text = NUMBERED.sub(lambda m: f"{m[0]} {ALIASES.get(m[1], m[1])}{m[2]}", normal(text))
    return [ALIASES.get(term, term) for term in re.findall(r"[a-z0-9]+", text)
            if term not in STOPWORDS and (len(term) >= 2 or term.isdigit())]


def mission_keys(text):
    return {ALIASES.get(name, name) + number for name, number in NUMBERED.findall(normal(text))}


@lru_cache(maxsize=2)
def build_index(_updated_at):
    records = json.loads(CORPUS.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1150, chunk_overlap=145, separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = []
    for record in records:
        for chunk in splitter.split_text(record["text"]):
            context = f'{record["mission"]} | {record["title"]}\n{chunk}'
            metadata = {key: record[key] for key in ("title", "mission", "kind", "url")}
            metadata["excerpt"] = chunk
            docs.append(Document(page_content=context, metadata=metadata))
    if not docs:
        raise ValueError("Corpus vacío")
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize)


def search(question, limit=6):
    index = build_index(CORPUS.stat().st_mtime_ns)
    terms = set(tokenize(question))
    if not terms:
        return []
    named = mission_keys(question)
    # Require a topical match as well as the named mission when the user includes a topic.
    generic = {"mission", "module", "spacecraft", "space", "used", "about", "engine"}
    mission_terms = named | {re.sub(r"\d+$", "", key) for key in named}
    topic = {term for term in terms - mission_terms - generic if not term.isdigit()}
    discriminating = {term for term in topic if term in index.vectorizer.idf}
    if topic and not discriminating:
        return []
    if not named and len(topic) > 1 and len(discriminating) < 2:
        return []
    if not topic and not named:
        return []

    scores = index.vectorizer.get_scores(sorted(terms))
    for i, doc in enumerate(index.docs):
        if doc.metadata["excerpt"].startswith(("Alternate Names:", "Facts in Brief:")):
            scores[i] *= 0.7
    ranked = []
    for raw_index in np.argsort(-scores, kind="stable"):
        i = int(raw_index)
        if scores[i] <= 0:
            break
        doc = index.docs[i]
        tags = mission_keys(doc.metadata["mission"])
        passage_terms = index.vectorizer.doc_freqs[i]
        if named and not (tags & named) and not named.intersection(passage_terms):
            continue
        if discriminating and not discriminating.intersection(passage_terms):
            continue
        ranked.append((doc, tags | named.intersection(passage_terms)))

    selected = []
    url_counts = Counter()

    def add(doc):
        url = doc.metadata["url"]
        if url_counts[url] < 2 and doc not in selected:
            selected.append(doc)
            url_counts[url] += 1

    if len(named) > 1:
        for key in sorted(named)[:limit]:
            for doc, tags in ranked:
                if key in tags:
                    add(doc)
                    break
    for unique in (True, False):
        for doc, _ in ranked:
            if len(selected) >= limit:
                break
            if not unique or not url_counts[doc.metadata["url"]]:
                add(doc)

    return [
        {"id": f"F{number}", **doc.metadata}
        for number, doc in enumerate(selected, 1)
    ]
