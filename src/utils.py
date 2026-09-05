def build_prompt(user_prompt: str, functions_text: str) -> str:
    """Build a chat-ML formatted prompt for function-call generation.

    Constructs a prompt string using the ``<|im_start|>`` /
    ``<|im_end|>`` chat-ML delimiters.  The *system* turn describes
    the assistant role and lists the available functions, while the
    *user* turn contains the actual user query.

    Args:
        user_prompt: The natural-language request from the user.
        functions_text: A JSON-formatted string describing the
            available functions and their parameters.

    Returns:
        The fully assembled prompt string ready to be tokenised and
        fed to the language model.
    """
    return (
        "<|im_start|>system\nYou are a function calling assistant.\n"
        f"Available functions:\n{functions_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n"
        "Respond only with a JSON object with keys 'name' "
        "and 'parameters'.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
