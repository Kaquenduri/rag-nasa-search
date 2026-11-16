# SOMA · Biblioteca de misiones

Servicio Django independiente para consultar antecedentes históricos de misiones,
naves, experimentos e instrumentos. El catálogo se prepara una vez y se puede
consultar aunque NASA o el túnel de Ollama estén temporalmente fuera de línea.
Django no necesita migraciones ni una base de datos para esta biblioteca.

## Arranque local

Desde esta carpeta:

```sh
python3 -m pip install --target .deps -r requirements.txt
export PYTHONPATH="$PWD/.deps"
python3 manage.py check
python3 manage.py runserver 127.0.0.1:8010
```

El corpus inicial viene incluido. Para actualizar las copias de las páginas
oficiales y regenerar el corpus:

```sh
PYTHONPATH=.deps python3 -m mission_rag.ingest --refresh
```

Sin `--refresh` se usan las copias locales disponibles y se reintentan las que
faltan. `data/ingest-report.json` enumera la cobertura e incidencias.
`data/raw/` contiene las copias HTML/XHTML y queda fuera de Git.
El catálogo se recorre desde una selección de misiones hacia sus naves e
instrumentos. Incluye páginas históricas de NASA sobre herramientas y
caminatas espaciales. Los archivos científicos binarios y los inventarios
operativos actuales quedan fuera de la cobertura.

## Ollama

La URL predeterminada es
`https://scouting-flatten-arrival.ngrok-free.dev/api/generate`.
El modelo de texto predeterminado es `qwen3.5:2b`. En el equipo que ejecuta Ollama:

```sh
ollama pull qwen3.5:2b
ollama list
```

Mantén ese equipo y el túnel activos. Si el modelo instalado tiene otro
nombre, configura `OLLAMA_MODEL` con el valor de `ollama list`. Para ejecutar
Django en la misma computadora que Ollama, puedes usar:

```sh
export OLLAMA_URL=http://127.0.0.1:11434/api/generate
export OLLAMA_MODEL=qwen3.5:2b
```

## Consulta y recuperación

Sin contactar al modelo:

```sh
PYTHONPATH=.deps python3 manage.py mission_query \
  "¿Para qué se usó el altímetro láser del Apollo 16?" --search-only
```

Con respuesta del modelo:

```sh
curl -sS http://127.0.0.1:8010/api/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question":"¿Para qué se usó el altímetro láser del Apollo 16?"}'
```

- `GET /healthz/` indica si el corpus está presente.
- `POST /api/search/` recibe el mismo JSON y devuelve hasta seis pasajes con
  título, misión, tipo, URL e ID de cita. Sirve para añadir evidencia a otro modelo.
- `POST /api/ask/` agrega la respuesta en español, con referencias `[F1]`,
  `[F2]` y sus URLs. Si no hay evidencia suficiente, devuelve `supported: false`.
  Si Ollama falla, devuelve HTTP 502 con los pasajes recuperados.

La búsqueda usa LangChain y BM25 con normalización y un vocabulario bilingüe
inicial. Funciona sin un modelo de embeddings y favorece los identificadores
exactos de misión e instrumento. Es posible que consultas en español con
sinónimos ausentes del vocabulario necesiten búsqueda semántica en una fase
posterior.

Para conectarlo más adelante a Swift, envía únicamente la pregunta histórica.
Si se usa como contexto de otro modelo, conserva los fragmentos, las citas
y la distinción entre precedentes e ideas de simulación. No mezcles los
pasajes históricos con la ficha clínica ni la telemetría.

El servidor de desarrollo escucha en loopback. Para exponer una prueba con un
servidor HTTPS, configura `DJANGO_ALLOWED_HOSTS`, `DJANGO_SECRET_KEY` y
`RAG_API_TOKEN`. El cliente puede enviar `Authorization: Bearer <token>`.

## Colección de investigación sobre salud en el espacio

El [portal BPS de NASA](https://science.nasa.gov/biological-physical/data/) dirige al
[OSDR](https://science.nasa.gov/biological-physical/data/osdr/) para datos de
investigación y a la [NASA Space Life Sciences Library](https://public.ksc.nasa.gov/nslsl/)
para literatura. La selección de `health_sources.json` recoge estudios humanos y
análogos terrestres, artículos de revista de PubMed Central y los PDF públicos
del [Human Research Program](https://humanresearchroadmap.nasa.gov/evidence/).

- `data/raw/research/` guarda PDF, XML JATS y registros originales de DOI. Está fuera de Git, igual que los otros originales.
- `data/research-corpus.json` contiene los textos extraídos. Se conserva el idioma original, sin traducciones, resúmenes ni inferencias. Para los PDF se usa su capa de texto por página. Para los artículos se extraen los párrafos del resumen y cuerpo por sección. Para OSDR se guardan las descripciones de los registros DOI, con sus licencias.
- `data/research-report.json` documenta la URL, huella SHA-256 del original y exclusiones por licencia o disponibilidad. Las figuras, tablas, anexos y archivos de los participantes se consultan en la fuente cuando corresponda. El texto de PDF puede heredar cortes de línea y orden de columnas de su capa de texto.

La colección se indexa como **evidencia de investigación**. No forma parte de
los procedimientos o del inventario de una misión concreta. El endpoint solo
devuelve los pasajes originales y sus referencias. No ejecuta Ollama.

```sh
curl -sS http://127.0.0.1:8010/api/research/search/ \
  -H 'Content-Type: application/json' \
  -d '{"question":"intolerancia ortostática y presión arterial"}'
```

La búsqueda devuelve hasta tres fragmentos de 800 caracteres cada uno.
Cada resultado incluye el `excerpt`, `source_id`, tipo de documento, URL y
página o sección, además del DOI o licencia cuando la fuente dispone de ellos.
Los identificadores `S1`, `S2`, etc. pertenecen a la consulta actual. Si el
servicio configura `RAG_API_TOKEN`, también se aplica a esta ruta.

Para actualizar esta selección y recuperar los originales:

```sh
python3 -m pip install --target .deps -r requirements-research.txt
PYTHONPATH=.deps python3 -m mission_rag.research_ingest --refresh
```

La búsqueda en Render solo necesita el JSON preparado. Las dependencias de
extracción de PDF y los archivos originales solo son necesarios para regenerarlo.

## Búsqueda híbrida con embeddings

Django construye el índice BM25 al arrancar la búsqueda y combina su ranking
con el ranking semántico cuando el índice vectorial está disponible. La
colección de salud sigue entregando hasta 3 extractos de 800 caracteres.

Los embeddings usan `GoogleGenerativeAIEmbeddings` de LangChain con el modelo
remoto `gemini-embedding-001`, que tiene una capa gratuita sujeta a límites.
Necesitas una clave de la API de Gemini creada en Google AI Studio. Configura en
**Render → Environment**:

| Variable | Valor para SOMA |
| --- | --- |
| `GOOGLE_API_KEY` | Tu clave de la API de Gemini (secreta) |
| `GOOGLE_EMBED_MODEL` | `gemini-embedding-001` (opcional; valor predeterminado) |
| `RAG_EMBED_MIN_SCORE` | `0.25` (opcional; ajustable según resultados) |

En Terminal, define la clave antes de generar el índice:

```sh
export GOOGLE_API_KEY=tu_clave
PYTHONPATH=.deps python3 manage.py build_embeddings
```

También puedes ejecutar el comando con `--collection missions` o
`--collection research`. El comando reutiliza los lotes guardados si se
interrumpe. Reejecútalo si actualizas el corpus o cambias de modelo. El índice
anterior de Ollama no es compatible: regenera los vectores y publica los
archivos nuevos antes de activar la búsqueda semántica en Render.

Sube `data/missions-vectors.npz` y `data/research-vectors.npz` a GitHub junto
a sus corpus. Cada archivo contiene vectores comprimidos de 256 dimensiones
y una huella que permite verificar que pertenecen a esa versión del texto.
Render carga los archivos y calcula similitud coseno en memoria. No necesita
una base vectorial ni descargar el corpus en cada build. Sin un índice
válido, si falta la clave o si Gemini no responde,
se usa BM25 con el mismo contrato de la API.

La consulta de búsqueda se envía a Gemini para obtener su embedding. Usa
términos de búsqueda sin nombres, fichas clínicas ni telemetría.
