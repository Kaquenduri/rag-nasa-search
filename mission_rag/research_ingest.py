"""Build the spaceflight health research corpus from selected official records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from time import sleep
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree

from .network import https_handler

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "health_sources.json"
DATA = ROOT / "data"
RAW = DATA / "raw" / "research"
HOSTS = {"humanresearchroadmap.nasa.gov", "pmc.ncbi.nlm.nih.gov", "api.datacite.org",
         "science.nasa.gov", "public.ksc.nasa.gov"}
PMC = "https://pmc.ncbi.nlm.nih.gov/"
LICENSE_URL = re.compile(r"https?://creativecommons\.org/(?:licenses/by/[1-4]\.0|publicdomain/zero/1\.0)(?=[/\s<]|$)")


def source_url_allowed(url):
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.hostname in HOSTS


class OfficialSourceRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not source_url_allowed(newurl):
            raise ValueError("Redirección fuera de las fuentes autorizadas.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, name, refresh):
    if not source_url_allowed(url):
        raise ValueError("Dominio de origen no admitido.")
    path = RAW / name
    if refresh or not path.is_file():
        request = Request(url, headers={"User-Agent": "SOMA-SpaceflightResearch/1.0",
                                        "Accept": "application/pdf, application/json, application/xml, text/html"})
        with build_opener(OfficialSourceRedirect, https_handler()).open(request, timeout=24) as response:
            payload = response.read(24_000_001)
        if len(payload) > 24_000_000:
            raise ValueError("Archivo supera 24 MB.")
        if not payload:
            raise ValueError("Respuesta vacía.")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    payload = path.read_bytes()
    return payload, {
        "retrieved_from": url,
        "snapshot_file": f"research/{path.name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def xml_text(element):
    return "".join(element.itertext()).strip() if element is not None else ""


def load_report(seed, config, refresh):
    from pypdf import PdfReader

    url = urljoin(config["reports_index"], seed["path"])
    payload, provenance = fetch(url, seed["id"] + ".pdf", refresh)
    if not payload.startswith(b"%PDF-"):
        raise ValueError("El archivo de NASA no es un PDF.")
    pdf = PdfReader(BytesIO(payload))
    sections = []
    unreadable = []
    for number, page in enumerate(pdf.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            sections.append({"page": number, "text": content})
        else:
            unreadable.append(number)
    if not sections:
        raise ValueError("El informe carece de una capa de texto.")
    return {
        **provenance, "id": seed["id"], "title": seed["title"], "kind": "informe técnico",
        "publisher": "NASA Human Research Program", "url": url,
        "topic": seed["topic"], "setting": "informe de evidencia",
        "rights": "Texto de informe público del NASA Human Research Program. Se conservan las atribuciones de terceros del original.",
        "page_count": len(pdf.pages), "pages_without_text": unreadable, "sections": sections,
    }


def article_sections(node, headings=()):
    if title := xml_text(node.find("title")):
        headings = (*headings, title)
    paragraphs = []
    for child in node:
        if child.tag == "p" and (paragraph := xml_text(child)):
            paragraphs.append(paragraph)
        elif child.tag == "sec":
            if paragraphs:
                yield {"section": " / ".join(headings), "text": "\n\n".join(paragraphs)}
                paragraphs = []
            yield from article_sections(child, headings)
    if paragraphs:
        yield {"section": " / ".join(headings), "text": "\n\n".join(paragraphs)}


def load_article(seed, _config, refresh):
    pmcid = seed["id"]
    url = PMC + "articles/" + pmcid + "/"
    api = PMC + "api/oai/v1/mh/?" + urlencode({
        "verb": "GetRecord", "identifier": f"oai:pubmedcentral.nih.gov:{pmcid[3:]}",
        "metadataPrefix": "pmc",
    })
    payload, provenance = fetch(api, pmcid + ".xml", refresh)
    root = ElementTree.fromstring(payload)
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    article = root.find(".//article")
    if article is None:
        raise ValueError("PMC no ofrece una copia de texto completo reutilizable en su API.")
    front = article.find("front/article-meta")
    if front is None:
        raise ValueError("El artículo carece de metadatos JATS.")
    license_nodes = front.findall("./permissions/license")
    rights = [xml_text(license_node) for license_node in license_nodes]
    # Evaluate actual license declarations, never the article body or incidental citations.
    for node in license_nodes:
        rights.extend(attr for el in node.iter() for key, attr in el.attrib.items()
                      if key.rsplit("}", 1)[-1] == "href")
    license_match = LICENSE_URL.search("\n".join(rights))
    if not license_match:
        # A few journals use an official license landing page instead of the CC URL.
        if not any(re.search(r"Creative Commons Attribution 4\.0 International License", line, re.I)
                   and not re.search(r"non.?commercial|no.?derivatives", line, re.I) for line in rights):
            raise ValueError("Licencia CC BY o CC0 no confirmada. Texto excluido.")
        license_url = "https://creativecommons.org/licenses/by/4.0/"
    else:
        license_url = license_match.group().replace("http://", "https://").rstrip("/") + "/"
    paper_title = xml_text(front.find("title-group/article-title"))
    abstract = front.find("abstract")
    body = article.find("body")
    sections = list(article_sections(abstract, ("Abstract",))) if abstract is not None else []
    if body is not None:
        sections.extend(article_sections(body))
    if not paper_title or not sections:
        raise ValueError("Artículo sin título o texto de sus secciones.")
    authors = []
    for author in front.findall("contrib-group/contrib[@contrib-type='author']"):
        name = " ".join(filter(None, (xml_text(author.find("name/given-names")),
                                      xml_text(author.find("name/surname")))))
        if name:
            authors.append(name)
    date = front.find("pub-date")
    return {
        **provenance, "id": pmcid, "title": paper_title, "kind": "artículo científico",
        "journal": xml_text(article.find("front/journal-meta/journal-title-group/journal-title")),
        "authors": authors, "year": xml_text(date.find("year")) if date is not None else "",
        "doi": xml_text(front.find("article-id[@pub-id-type='doi']")), "url": url,
        "license_url": license_url, "rights": xml_text(license_nodes[0]) if license_nodes else "",
        "topic": seed["topic"], "setting": seed["setting"], "sections": sections,
    }


def load_study(seed, _config, refresh):
    identifier = seed["id"]
    url = "https://osdr.nasa.gov/bio/repo/data/studies/" + identifier
    payload, provenance = fetch("https://api.datacite.org/dois/" + seed["doi"], identifier + ".json", refresh)
    record = json.loads(payload)["data"]["attributes"]
    if record["url"].rstrip("/") != url or record["types"]["resourceTypeGeneral"] != "Dataset":
        raise ValueError("El DOI no corresponde al estudio OSDR indicado.")
    license_entry = next((item for item in record.get("rightsList", [])
                          if (item.get("rightsIdentifier") or "").lower() in {"cc0-1.0", "cc-by-4.0"}), None)
    if license_entry is None:
        raise ValueError("Licencia CC BY o CC0 no confirmada. Metadatos excluidos.")
    sections = [{"section": desc.get("descriptionType", "Description"), "text": desc["description"].strip()}
                for desc in record["descriptions"]
                if desc.get("descriptionType") in {"Abstract", "Other"} and desc.get("description", "").strip()]
    if not sections:
        raise ValueError("El DOI carece de descripción reutilizable.")
    return {
        **provenance, "id": identifier, "title": record["titles"][0]["title"],
        "kind": "ficha de estudio", "url": url, "doi": seed["doi"],
        "publisher": record["publisher"], "year": record.get("publicationYear"),
        "authors": [creator["name"] for creator in record.get("creators", [])],
        "license_url": license_entry["rightsUri"], "rights": license_entry["rights"],
        "topic": seed["topic"], "setting": seed["setting"], "sections": sections,
    }


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ingest(refresh=False):
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    seeds = [(loader, seed) for key, loader in (
        ("reports", load_report), ("articles", load_article), ("studies", load_study)
    ) for seed in config[key]]

    def download(item):
        loader, seed = item
        try:
            document = loader(seed, config, refresh)
            print(f'{seed["id"]}: {len(document["sections"])} páginas o secciones', flush=True)
            return document, None
        except (OSError, ValueError, KeyError, IndexError, ElementTree.ParseError) as exc:
            print(f'{seed["id"]}: {exc}', flush=True)
            return None, {"id": seed["id"], "error": str(exc)}

    # PMC requests are serialized with spacing. NASA and DataCite use a small separate pool.
    other = [item for item in seeds if item[0] is not load_article]
    articles = [item for item in seeds if item[0] is load_article]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, item) for item in other]
        results = []
        for item in articles:
            _, seed = item
            will_fetch = refresh or not (RAW / f'{seed["id"]}.xml').exists()
            results.append(download(item))
            if will_fetch:
                sleep(0.6)
        results.extend(future.result() for future in futures)

    failures = [error for _, error in results if error is not None]
    records = [record for record, _ in results if record is not None]
    if not records:
        raise SystemExit("No se obtuvo ningún documento. Se conserva la colección previa.")
    write_json(DATA / "research-corpus.json", records)
    write_json(DATA / "research-report.json", {
        "origin": config["origin"], "reports_index": config["reports_index"],
        "journal_index": config["journal_index"], "documents": len(records),
        "sources": [{key: value for key, value in record.items() if key not in {"sections", "topic"}}
                    for record in records],
        "excluded_or_unavailable": failures,
    })
    print(f"{len(records)} documentos guardados; {len(failures)} exclusiones o fallos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Volver a descargar los originales.")
    ingest(parser.parse_args().refresh)
