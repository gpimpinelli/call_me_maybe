import numpy as np

def soft_max(logits: list[float]) -> np.ndarray:
    np_logits = np.array(logits)
    max_logit = np.max(np_logits)
    new_logits = np.exp(np_logits - max_logit)
    probability = new_logits / np.sum(new_logits)
    return probability

def get_logit_max(logits: np.ndarray) -> int:
    return int(np.argmax(logits))

def build_prompt(user_prompt: str, functions_text: str) -> str:
    return (
        f"You are a function calling assistant.\n"
        f"Available functions:\n{functions_text}\n\n"
        f"User request: {user_prompt}\n"
        f"Respond only with a JSON object with keys 'name' and 'parameters'.\n"
    )