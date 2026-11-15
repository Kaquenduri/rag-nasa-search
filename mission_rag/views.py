import json
from secrets import compare_digest

from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

from .generation import GeneratorError, ask
from .research_retrieval import search_research
from .retrieval import CORPUS, search


@require_GET
def health(_request):
    ready = CORPUS.is_file()
    return JsonResponse({"status": "ready" if ready else "missing_corpus"}, status=200 if ready else 503)


def authorized(request):
    token = settings.RAG_API_TOKEN
    header = request.headers.get("Authorization", "")
    return not token or compare_digest(header.encode(), f"Bearer {token}".encode())


def read_question(request):
    if request.content_type != "application/json":
        return None, JsonResponse({"error": "Envía Content-Type application/json."}, status=415)
    try:
        data = json.loads(request.body)
    except RequestDataTooBig:
        return None, JsonResponse({"error": "Solicitud demasiado grande."}, status=413)
    except (ValueError, UnicodeError):
        return None, JsonResponse({"error": "JSON inválido."}, status=400)
    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        return None, JsonResponse({"error": "question debe contener entre 1 y 2000 caracteres."}, status=400)
    return question.strip(), None


def evidence_for(request, retrieve=search):
    if not authorized(request):
        return None, [], JsonResponse({"error": "Acceso no autorizado."}, status=401)
    question, error = read_question(request)
    if error is not None:
        return None, [], error
    try:
        return question, retrieve(question), None
    except (OSError, ValueError):
        return None, [], JsonResponse({"error": "Corpus no disponible. Ejecuta la importación."}, status=503)


@require_POST
def search_sources(request):
    _, evidence, error = evidence_for(request)
    return error if error is not None else JsonResponse({"sources": evidence})


@require_POST
def search_research_sources(request):
    _, evidence, error = evidence_for(request, search_research)
    if error is not None:
        return error
    return JsonResponse({"collection": "spaceflight_health_research", "sources": evidence})


@require_POST
def ask_question(request):
    question, evidence, error = evidence_for(request)
    if error is not None:
        return error
    if not evidence:
        return JsonResponse({"answer": "No encontré evidencia suficiente en el corpus para responder.",
                             "supported": False, "sources": []})
    try:
        return JsonResponse(ask(question, evidence))
    except GeneratorError as exc:
        return JsonResponse({"error": str(exc), "sources": evidence}, status=502)
