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
