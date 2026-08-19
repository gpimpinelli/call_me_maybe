from pydantic impoer BaseModel
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedDecoder(BaseModel):
    model: Small_LLM_Model
    tokenizer: CustomTokenizer
    schema: SchemaAwareValidator