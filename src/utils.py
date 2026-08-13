import numpy as np

def soft_max(logits: list[float]) -> np.ndarray:
    np_logits = np.array(logits)
    max_logit = np.max(np_logits)
    new_logits = exp(np_logits - max_logit)
    probability = new_logits / np.sum(new_logits)
    return probability

def get_logit_max(logits: np.ndarray) -> float:
    return np.max(logits)