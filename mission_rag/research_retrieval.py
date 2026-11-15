from collections import Counter
from functools import lru_cache
import json
import re

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .retrieval import CORPUS, normal, tokenize
from .vector_search import hybrid_rank

RESEARCH_CORPUS = CORPUS.with_name("research-corpus.json")
KEYWORDS = {
    "mareo": "motion sickness dizziness vestibular",
    "mareado": "motion sickness dizziness vestibular",
    "mareos": "motion sickness dizziness vestibular",
    "vertigo": "vertigo vestibular dizziness",
    "nausea": "nausea motion sickness",
    "nauseas": "nausea motion sickness",
    "equilibrio": "balance vestibular postural",
    "ortostatica": "orthostatic",
    "ortostatico": "orthostatic",
    "intolerancia": "intolerance",
    "presion arterial": "blood pressure",
    "tension arterial": "blood pressure",
    "desmayo": "syncope presyncope orthostatic",
    "cardiaco": "cardiac cardiovascular",
    "cardiovascular": "cardiovascular",
    "ritmo circadiano": "circadian rhythm",
    "sueno": "sleep",
    "cansancio": "fatigue alertness",
    "fatiga": "fatigue",
    "dormir": "sleep",
    "alerta": "alertness",
    "densidad osea": "bone density",
    "perdida osea": "bone loss",
    "hueso": "bone",
    "huesos": "bone",
    "fractura": "fracture",
    "fracturas": "fracture",
    "atrofia muscular": "muscle atrophy",
    "muscular": "muscle",
    "ejercicio": "exercise",
    "vision": "vision visual ocular",
    "neuroocular": "neuro ocular SANS",
    "neuro ocular": "neuro ocular SANS",
    "oxigeno": "oxygen",
    "hipoxia": "hypoxia",
    "descompresion": "decompression",
    "deshidratacion": "dehydration",
    "hidratacion": "hydration",
    "nutricion": "nutrition",
    "polvo lunar": "lunar dust",
    "respiratorio": "respiratory",
    "sintomas": "symptoms",
    "inflamacion": "inflammation",
    "reposo": "bed rest",
}
GENERIC = set("""
health salud humano humanos human humans astronaut astronauta astronautas astronauts
space espacio flight spaceflight microgravity microgravedad crew mission informe evidence
report durante despues antes vuelo busca aplica fuentes estudios estudio revisiones sintomas
symptoms tripulantes investigacion investigador paper articulo cita articles evidence
no hacer hago hacerme haria puedo podria puede podrian estaba estoy siente siento
sientan hace hola ayuda decir quieres quiero saber sabes recomendar recomienda informacion
dame recomiendas recomiende padezco presenta asociados hola investigaciones documentos
trabajos hallazgos observaciones provoca causa descrito experimentos acerca provoca
""".split())


def research_terms(question):
    query = " ".join(re.findall(r"[a-z0-9]+", normal(question)))
    expanded = [
        terms for phrase, terms in KEYWORDS.items() if f" {phrase} " in f" {query} "
    ]
    # Keep provenance text intact. Expansion is applied only to the search query.
    return set(tokenize(" ".join([query, *expanded]))) - GENERIC


@lru_cache(maxsize=2)
def research_index(_mtime):
    records = json.loads(RESEARCH_CORPUS.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = []
    for record in records:
        metadata = {key: record[key] for key in (
            "id", "title", "kind", "url", "setting", "doi", "journal", "publisher",
            "authors", "year", "license_url", "rights",
        ) if key in record}
        metadata["source_id"] = metadata.pop("id")
        for part in record["sections"]:
            locator = {key: part[key] for key in ("page", "section") if key in part}
            if "page" in locator:
                locator["url"] = f'{record["url"]}#page={locator["page"]}'
            for excerpt in splitter.split_text(part["text"]):
                docs.append(Document(
                    page_content=excerpt,
                    metadata={**metadata, **locator, "excerpt": excerpt},
                ))
    if not docs:
        raise ValueError("Colección de investigación vacía.")
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize)


def search_research(question, limit=3):
    terms = research_terms(question)
    if not terms:
        return []
    index = research_index(RESEARCH_CORPUS.stat().st_mtime_ns)
    scores = index.vectorizer.get_scores(sorted(terms))
    lexical = [int(i) for i in np.argsort(-scores, kind="stable")
               if scores[i] > 0 and terms.intersection(index.vectorizer.doc_freqs[i])]
    ranked = [index.docs[i] for i in hybrid_rank("research", index.docs, question, lexical)]
    selected = []
    source_counts = Counter()
    for maximum in (1, 2):
        for doc in ranked:
            if len(selected) >= limit:
                break
            source = doc.metadata["source_id"]
            if source_counts[source] < maximum and doc not in selected:
                selected.append(doc)
                source_counts[source] += 1
    return [{"id": f"S{number}", **doc.metadata} for number, doc in enumerate(selected, 1)]
