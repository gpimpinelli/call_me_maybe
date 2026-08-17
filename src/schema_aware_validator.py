from typing import Any
from pydantic import BaseModel, PrivateAttr

class SchemaAwareValidator(BaseModel):
    json_functions: list[dict]
    function_by_name: dict[str, dict] = PrivateAttr(default_factory=dict)
    name_trie: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        self.function_by_name = {f["name"]: f for f in self.json_functions}
        self.name_trie = self._init_trie([f["name"] for f in self.json_functions])
    
    def _init_trie(self, words: list[str]):
        trie: dict = {}
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node[None] = word
        return trie

    def next_char(self, partial: str) -> set[str]:
        node = self.name_trie 
        for char in partial:
            if char not in node:
                return set()
            node = node[char]
        return {c for c in node if c is not None}

    def is_complete(self, partial: str) -> bool:
        node = self.name_trie
        for char in partial:
            if char not in node:
                return False
            node = node[char]
        return None in node

    def param_trie_for(self, function_name: str) -> dict:
        params = self.function_by_name.get(function_name, {}).get("parameters", {})
        return self._init_trie(list(params.keys()))
