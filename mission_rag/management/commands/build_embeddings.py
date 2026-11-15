from django.core.management.base import BaseCommand, CommandError

from mission_rag.research_retrieval import RESEARCH_CORPUS, research_index
from mission_rag.retrieval import CORPUS, build_index
from mission_rag.vector_search import build_vectors, embeddings_configured


class Command(BaseCommand):
    help = "Precalcula los embeddings para los corpus de SOMA y guarda sus índices."

    def add_arguments(self, parser):
        parser.add_argument("--collection", choices=("missions", "research", "all"), default="all")

    def handle(self, collection, **options):
        if not embeddings_configured():
            raise CommandError("Define OLLAMA_EMBED_URL y OLLAMA_EMBED_MODEL antes de generar el índice.")

        collections = {"missions": (CORPUS, build_index), "research": (RESEARCH_CORPUS, research_index)}
        for name, (path, load) in collections.items():
            if collection not in ("all", name):
                continue
            try:
                result = build_vectors(name, load(path.stat().st_mtime_ns).docs, self.stdout.write)
            except (OSError, ValueError, KeyError) as exc:
                raise CommandError(f"{name}: {exc}. El progreso guardado se reutiliza al reintentar.") from exc
            self.stdout.write(self.style.SUCCESS(str(result)))
