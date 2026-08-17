from pydantic import BaseModel, Field, model_validator, PrivateAttr
import re
from typing import Any, ClassVar


class CustomTokenizer(BaseModel):
    ## Per bonus capire come fare un traduttore di regex
    SPLIT_REGEX: ClassVar[re.Pattern] = re.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
    r"""|[^\r\n\w]?[^\W\d_]+"""
    r"""|\d"""
    r"""|\ ?[^\s\w]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+""",
    re.VERBOSE
    )
    token_id: dict[str, int]
    dict_merge: dict[tuple[str, str], int]
    special_token_id: dict[str, Any]
    _id_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _special_id_token: dict[int, Any] = PrivateAttr(default_factory=dict)
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _special_split_regex: re.Pattern | None = PrivateAttr(default=None)
    trie_token: dict[str, Any] = Field(default_factory=dict)
            
    @model_validator(mode='before')
    @classmethod
    def filter_special_token(cls, data):
        if isinstance(data, dict):
            special_token_id = data.get('special_token_id', None)
            if special_token_id:
                special_token_id = {token['content']: token['id'] for token in special_token_id}
                data['special_token_id'] = special_token_id
        return data

    def model_post_init(self, __context) -> None:
        self._id_token = {id_: token for token, id_ in self.token_id.items()}
        self._byte_encoder = self._bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._special_id_token = {v: k for k, v in self.special_token_id.items()}
        if self.special_token_id:
            pattern = "|".join(re.escape(tok) for tok in self.special_token_id)
            self._special_split_regex = re.compile(f"({pattern})")
        self.trie_token = self._init_trie()

    @property
    def id_token(self):
        return self._id_token

    @property
    def special_id_token(self):
        return self._special_id_token

    def _init_trie(self):
        trie: dict = {}
        for id_ in self._id_token:
            if id_ in self.special_id_token:
                continue
            try:
                decoded = self.decode(id_)
            except (KeyError, UnicodeDecodeError):
                continue
            if not decoded or "\ufffd" in decoded:
                continue
            node = trie
            for char in decoded:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(id_)
        return trie

    def _bytes_to_unicode(self):
        tokens = (
            list(range(ord('!'), ord('~') + 1)) +
            list(range(ord('¡'), ord('¬') + 1)) +
            list(range(ord('®'), ord('ÿ') + 1))
        )
        copy = tokens[:]
        n = 0
        for c in range(256):
            if c not in tokens:
                copy.append(c)
                tokens.append(256 + n)
                n += 1
        return dict(zip(copy, [chr(c) for c in tokens]))

    def encode(self, text: str) -> list[int]:
        l_index = []
        pieces = self._special_split_regex.split(text) if self._special_split_regex else [text]
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_token_id:
                l_index.append(self.special_token_id[piece])
                continue
            l_text = self.SPLIT_REGEX.findall(piece)
            for word in l_text:
                word_bytes = word.encode("utf-8")
                chars = [self._byte_encoder[b] for b in word_bytes]
                while len(chars) >= 2:
                    best_pair = None
                    best_rank = float('inf')
                    for i in range(len(chars) - 1):
                        pair = (chars[i], chars[i + 1])
                        rank = self.dict_merge.get(pair)
                        if rank is not None and rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                    if best_pair is None:
                        break
                    new_word = []
                    i = 0
                    while i < len(chars):
                        if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                            new_word.append(chars[i] + chars[i + 1])
                            i += 2
                        else:
                            new_word.append(chars[i])
                            i += 1
                    chars = new_word
                for c in chars:
                    l_index.append(self.token_id[c])
        return l_index

    def decode(self, id_: int) -> str:
        if id_ in self.special_id_token:
            return self.special_id_token[id_]
        token_str = self._id_token.get(id_, "")
        byte_list = [self._byte_decoder[c] for c in token_str]
        return bytes(byte_list).decode("utf-8", errors="replace")

    def decode_list(self, ids: list[int]) -> str:
        res = ""
        accumulate_byte = bytearray()
        for id_ in ids:
            if id_ in self.special_id_token:
                if accumulate_byte:
                    res += accumulate_byte.decode("utf-8", errors="replace")
                    accumulate_byte.clear()
                res += self.special_id_token[id_]
            else:
                token_str = self._id_token.get(id_, "")
                for c in token_str:
                    accumulate_byte.append(self._byte_decoder[c])
        if accumulate_byte:
            res += accumulate_byte.decode("utf-8", errors="replace")
        return res