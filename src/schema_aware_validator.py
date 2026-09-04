from typing import Any
from pydantic import BaseModel, PrivateAttr


def validate_functions_definition(json_functions: list) -> None:
    if not isinstance(json_functions, list):
        raise ValueError("The functions JSON must be a list of objects.")

    for i, func in enumerate(json_functions):
        if "name" not in func:
            raise ValueError(
                f"Validation error: the function at index {i} "
                "is missing the 'name' field."
            )

        func_name = func["name"]

        if "description" not in func:
            raise ValueError(
                f"Validation error: the function '{func_name}'"
                " is missing the 'description' field."
            )

        if "parameters" not in func:
            raise ValueError(
                f"Validation error: the function '{func_name}'"
                " is missing the 'parameters' field."
            )

        parameters = func["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Validation error: the 'parameters' of '{func_name}'"
                " must be a dictionary (object)."
            )


class SchemaAwareValidator(BaseModel):
    json_functions: list[dict]
    _function_by_name: dict[str, dict] = PrivateAttr(default_factory=dict)
    _name_trie: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        validate_functions_definition(self.json_functions)
        self._function_by_name = {f["name"]: f for f in self.json_functions}
        self._name_trie = self._init_trie(
            [f["name"] for f in self.json_functions]
            )

    @property
    def name_trie(self) -> dict[str, Any]:
        return self._name_trie

    @property
    def function_by_name(self) -> dict[str, dict]:
        return self._function_by_name

    def _init_trie(self, words: list[str]) -> dict:
        trie: dict = {}
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node[None] = word
        return trie

    @staticmethod
    def _walk_trie(trie: dict[str, Any], partial: str) -> set[str]:
        node = trie
        for char in partial:
            if char not in node:
                return set()
            node = node[char]
        return {c for c in node if c is not None}

    @staticmethod
    def _is_complete_in_trie(trie: dict[str, Any], partial: str) -> bool:
        node = trie
        for char in partial:
            if char not in node:
                return False
            node = node[char]
        return None in node

    @staticmethod
    def _unique_completion(trie: dict[str, Any], partial: str) -> str | None:
        """Se, dato il prefisso `partial`, esiste UN SOLO modo di completare la
        parola nel trie, ritorna il suffisso restante (senza includerlo il
        `partial` stesso). Altrimenti ritorna None (prefisso ambiguo, dead-end,
        o gia' completo con possibili continuazioni).
        Questo permette di "auto-completare" nomi di funzione/parametro senza
        dover interrogare il modello carattere per carattere quando il resto
        e' comunque forzato dallo schema.
        """
        node = trie
        for char in partial:
            if char not in node:
                return None
            node = node[char]

        suffix = ""
        while True:
            children = [c for c in node if c is not None]
            has_end = None in node
            if has_end and not children:
                return suffix
            if has_end and children:
                return None
            if len(children) == 1:
                char = children[0]
                suffix += char
                node = node[char]
                continue
            return None

    def next_char_for_name(self, partial: str) -> set[str]:
        return self._walk_trie(self.name_trie, partial)

    def is_complete(self, partial: str) -> bool:
        return self._is_complete_in_trie(self.name_trie, partial)

    def get_unique_completion_for_name(self, partial: str) -> str | None:
        return self._unique_completion(self.name_trie, partial)

    def get_param_type(self, function_name: str, param_name: str) -> Any:
        func = self._function_by_name.get(function_name, {})
        params = func.get("parameters", {})
        return params.get(param_name, {}).get("type", "string")

    def get_required_params(self, function_name: str) -> list[str]:
        func = self._function_by_name.get(function_name, {})
        return list(func.get("parameters", {}).keys())
