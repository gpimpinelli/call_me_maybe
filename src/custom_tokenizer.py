from pydantic import BaseModel, model_validator, Field, PrivateAttr

class CustomTokenizer(BaseModel):
    token_id: dict[str, int]
    _id_token: dict[int, str] = PrivateAttr(default_factory=dict)
    
    def model_post_init(self, __contex) -> None:
        self._id_token = {id_: token for token, id_ in self.token_id.items()}

    @property
    def id_token(self):
        return self._id_token

    def decode(self, id_: int) -> str:
        return self._id_token.get(id_, "").replace("Ġ", " ").replace("Ċ", "\n")

    def encode(self, token: str) -> list[int]:
        ...