import json

from django.core.management.base import BaseCommand, CommandError

from mission_rag.generation import GeneratorError, ask
from mission_rag.retrieval import search


class Command(BaseCommand):
    help = "Consulta local de la biblioteca. Usa --search-only para revisar los fragmentos."

    def add_arguments(self, parser):
        parser.add_argument("question")
        parser.add_argument("--search-only", action="store_true")

    def handle(self, question, search_only, **options):
        try:
            evidence = search(question)
        except (OSError, ValueError) as exc:
            raise CommandError("Corpus no disponible. Ejecuta la importación.") from exc
        if search_only or not evidence:
            result = {"sources": evidence}
        else:
            try:
                result = ask(question, evidence)
            except GeneratorError as exc:
                raise CommandError(str(exc)) from exc
        self.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
