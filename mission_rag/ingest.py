"""Download and extract the historic sources used by the mission library."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning
import warnings

from .network import https_handler

ROOT = Path(__file__).resolve().parent.parent / "data"
ARCHIVE = "https://nssdc.gsfc.nasa.gov"
LUNAR = ARCHIVE + "/planetary/lunar/"
CATALOG = ARCHIVE + "/planetary/projectdata.html"
APOLLO = (8, 10, 11, 12, 13, 14, 15, 16, 17)
MISSIONS = {
    "Cassini", "Huygens (Cassini Probe)", "Voyager 1", "Voyager 2",
    "Viking 1 Orbiter", "Viking 1 Lander", "Viking 2 Orbiter", "Viking 2 Lander",
    "Galileo", "Mariner 2", "Mariner 4", "Mariner 9", "Surveyor 1", "Surveyor 3",
    "Lunar Reconnaissance Orbiter", "Lunar Prospector", "Mars Global Surveyor",
    "Mars Pathfinder", "Mars Exploration Rover Spirit", "Mars Exploration Rover Opportunity",
    "Mars Reconnaissance Orbiter", "2001 Mars Odyssey", "Mars Pathfinder Rover", "Dawn", "Rosetta", "Stardust", "Hayabusa (Muses-C)",
}
EXTRAS = {
    LUNAR + "apollo.html": ("Programa Apollo", "Apollo", "programa"),
    LUNAR + "apollo_lrv.html": ("Vehículo lunar Apollo", "Apollo 15, Apollo 16, Apollo 17", "equipo"),
    LUNAR + "apolloland.html": ("Alunizajes Apollo", "Apollo", "programa"),
    ARCHIVE + "/planetary/chrono_astronaut.html": ("Cronología de astronautas", "Mercury, Gemini, Apollo", "cronología"),
    "https://www.nasa.gov/suits-and-rovers/spacewalking-tools/": (
        "Herramientas de caminatas espaciales", "Apollo 17, ISS", "equipo"
    ),
    "https://nasa.gov/wp-content/uploads/static/history/alsj/WOTM/WOTM-SCB.html": (
        "Bolsas de recolección de muestras", "Apollo 15, Apollo 16, Apollo 17", "equipo"
    ),
    "https://www.nasa.gov/skylab/": ("Skylab", "Skylab", "misión"),
    "https://www.nasa.gov/reference/international-space-station/": (
        "Referencia de la Estación Espacial Internacional", "ISS", "misión"
    ),
    "https://www.nasa.gov/history/space-station-20th-spacewalking-history/": (
        "Historia de las caminatas espaciales", "Gemini, Apollo, Skylab, ISS", "operaciones"
    ),
    "https://www.nasa.gov/history/skylab-3-spacewalks-and-science/": (
        "Caminatas y experimentos de Skylab 3", "Skylab 3", "operaciones"
    ),
    "https://www.nasa.gov/suits-and-rovers/spacesuits/": (
        "Trajes espaciales", "Apollo, ISS", "equipo"
    ),
    "https://science.nasa.gov/mission/cassini/": ("Cassini", "Cassini, Huygens", "misión"),
    "https://science.jpl.nasa.gov/projects/rosetta/": ("Rosetta", "Rosetta", "misión"),
    "https://science.nasa.gov/mission/hayabusa/": ("Hayabusa", "Hayabusa", "misión"),
    "https://science.nasa.gov/mission/odyssey/": ("Mars Odyssey", "2001 Mars Odyssey", "misión"),
    "https://science.nasa.gov/mission/msl-curiosity/": ("Curiosity", "Mars Science Laboratory", "misión"),
    "https://science.nasa.gov/mission/mars-2020-perseverance/": ("Perseverance", "Mars 2020", "misión"),
    ARCHIVE + "/nmc/spacecraft/display.action?id=1998-067A": (
        "Estación Espacial Internacional", "ISS", "nave"
    ),
    ARCHIVE + "/nmc/spacecraft/display.action?id=1973-027A": ("Skylab", "Skylab", "nave"),
}
NMC = re.compile(r"/nmc/(spacecraft|experiment)/display\.action$")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def nasa_url(url):
    parts = urlsplit(url)
    host = parts.hostname or ""
    return parts.scheme == "https" and (host == "nasa.gov" or host.endswith(".nasa.gov"))


class NASAOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not nasa_url(newurl):
            raise ValueError("Redirección fuera de NASA")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, refresh=False):
    if not nasa_url(url):
        raise ValueError("URL fuera de NASA")
    path = ROOT / "raw" / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".html")
    if refresh or not path.exists():
        request = Request(url, headers={"User-Agent": "SOMA-MissionLibrary/1.0 (research prototype)", "Accept": "text/html,application/xhtml+xml,text/xml"})
        with build_opener(NASAOnlyRedirect, https_handler()).open(request, timeout=14) as reply:
            if reply.headers.get_content_type() not in {"text/html", "application/xhtml+xml", "text/xml", "application/xml"}:
                raise ValueError("La respuesta no es una ficha HTML/XHTML")
            data = reply.read(3_000_001)
        if len(data) > 3_000_000:
            raise ValueError("Ficha supera el límite de 3 MB")
        path.write_bytes(data)
    return BeautifulSoup(path.read_bytes(), "html.parser"), path


def normalized_url(base, href):
    parts = urlsplit(urljoin(base, href))
    return parts._replace(scheme="https", fragment="").geturl()


def catalog_links(page, base, mission):
    for link in page.select("a[href]"):
        url = normalized_url(base, link["href"])
        parts = urlsplit(url)
        match = NMC.fullmatch(parts.path)
        identifier = parse_qs(parts.query).get("id", [""])[0]
        if parts.hostname == "nssdc.gsfc.nasa.gov" and match and re.fullmatch(r"[\w-]{3,32}", identifier):
            kind = "nave" if match[1] == "spacecraft" else "instrumento"
            name = clean(link.get_text(" ", strip=True)) or identifier
            yield ARCHIVE + parts.path + "?" + urlencode({"id": identifier}), (name, mission, kind)


def clean(text):
    return " ".join(text.split())


def extract(page, url, label, mission, kind):
    body = page.select_one("#rightcontent") or page.select_one("main, article") or page.body
    if body is None:
        raise ValueError("Ficha sin cuerpo")
    heading = body.find("h1")
    title = clean(heading.get_text(" ", strip=True)) if heading else label
    if title and not re.search(r"[\w]", title):
        title = label
    if not title or re.search(r"(not found|access denied|no results|server error)", title, re.I):
        raise ValueError("La ficha no está disponible")
    for comment in body.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for node in body.select("script,style,nav,footer,header,form,iframe,noscript,.usa-banner"):
        node.decompose()
    if NMC.fullmatch(urlsplit(url).path):
        description = body.select_one(".urone")
        if description is None and "an error has occurred" in body.get_text(" ", strip=True).lower():
            raise ValueError("El catálogo de NASA devolvió un error interno")
        if description is None or not description.find("p"):
            raise ValueError("El registro no contiene descripción")
        brief = body.select_one(".urtwo")
        parts = [description.get_text("\n", strip=True)]
        if brief:
            for tag in brief.find_all("h2"):
                if clean(tag.get_text()) in {"Facts in Brief", "Alternate Names"}:
                    sibling = tag.find_next_sibling()
                    if sibling:
                        parts.append(clean(tag.get_text()) + ": " + clean(sibling.get_text(" ", strip=True)))
        text = "\n".join(parts)
    else:
        for node in body.find_all("br"):
            node.replace_with("\n")
        text = body.get_text(" ", strip=False)
        # Keep row titles adjacent to their metadata on legacy Apollo pages.
        text = re.sub(r"\s*(Experiment Description|Data Sets|No Data Sets Available at NSSDCA\.)\s*", "\n", text)
    lines = [clean(line) for line in text.splitlines()]
    text = "\n".join(dict.fromkeys(line for line in lines if line)).strip()
    if len(text) < 70:
        raise ValueError("La ficha no contiene texto suficiente")
    return dict(title=title, mission=mission, kind=kind, url=url, text=text)


def ingest(refresh=False):
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    seeds = dict(EXTRAS)
    seeds.update({LUNAR + f"apollo{number}info.html": (f"Apollo {number}", f"Apollo {number}", "misión") for number in APOLLO})
    failures = []

    def record_failure(url, exc):
        failures.append({"url": url, "error": str(exc)[:250]})

    try:
        page, _ = fetch(CATALOG, refresh)
        for link in page.select("a[href]"):
            title = clean(link.get_text(" ", strip=True))
            if title in MISSIONS or title.startswith("Apollo "):
                url = normalized_url(CATALOG, link["href"])
                if nasa_url(url):
                    seeds[url] = (title, title, "nave" if "/nmc/spacecraft/" in url else "misión")
    except (OSError, ValueError) as exc:
        record_failure(CATALOG, exc)

    def download(entry):
        url, metadata = entry
        try:
            page, path = fetch(url, refresh)
            links = list(catalog_links(page, url, metadata[1]))
            doc = extract(page, url, *metadata)
            doc["snapshot_file"] = path.name
            doc["fetched_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            return url, doc, links, None
        except (OSError, ValueError) as exc:
            return url, None, [], exc

    sources = []
    visited = set()
    queue = seeds
    # Traverse mission -> spacecraft -> instrument, without following datasets or unrelated archives.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for level in range(3):
            candidates = {url: metadata for url, metadata in queue.items() if url not in visited}
            visited.update(candidates)
            print(f"Descargando o leyendo {len(candidates)} fichas del nivel {level + 1}…", flush=True)
            queue = {}
            for url, document, links, error in pool.map(download, candidates.items()):
                if error:
                    record_failure(url, error)
                    continue
                sources.append(document)
                queue.update(links)

    if not sources:
        raise SystemExit(f"No se obtuvo ninguna fuente. Primer error: {failures[0]['error']}")
    for name, value in (
        ("corpus.json", sources),
        ("ingest-report.json", {"sources": len(sources), "failures": failures,
                                "missions": sorted({doc["mission"] for doc in sources})}),
    ):
        output = ROOT / name
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(output)
    print(f"{len(sources)} fichas y {len(failures)} incidencias; consulta data/ingest-report.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Volver a descargar las fuentes.")
    ingest(parser.parse_args().refresh)
