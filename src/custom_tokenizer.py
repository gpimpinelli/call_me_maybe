from pydantic import BaseModel, model_validator, Field, PrivateAttr
import re
from typing import ClassVar

class CustomTokenizer(BaseModel):
    SPLIT_REGEX: ClassVar[re.Pattern] = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d|\s?[^\W\d_]+|\s?\d+|\s?[^\s\w]+|\s+(?!\S)|\s+""",
        re.VERBOSE
    )
    token_id: dict[str, int]
    dict_merge: dict[tuple[str, str], int]
    _id_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)    

    def model_post_init(self, __contex) -> None:
        self._id_token = {id_: token for token, id_ in self.token_id.items()}
        self._byte_encoder = self._bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

    @property
    def id_token(self):
        return self._id_token

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
        l_text = self.SPLIT_REGEX.findall(text)
        l_index = []
        for word in l_text:
            word_bytes = word.encode("utf-8")
            chars = [self._byte_encoder[b] for b in word_bytes]
            while len(chars) >= 2:
                best_pair = None
                best_rank = float('inf')
                for i in range(len(chars)-1):
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
        token_str = self._id_token.get(id_, "")
        byte_list = [self._byte_decoder[c] for c in token_str]
        return bytes(byte_list).decode("utf-8", errors="replace")

    def decode_list(self, ids: list[int]) -> str:
        text = "".join([self._id_token[id] for id in ids])
        byte_list = [self._byte_decoder[c] for c in text]
        return bytes(byte_list).decode("utf-8", errors="replace")
