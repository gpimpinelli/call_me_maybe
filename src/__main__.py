from typing import Any
import sys
import json
import argparse
import os
from .custom_tokenizer import CustomTokenizer
from .utils import build_prompt
from .schema_aware_validator import SchemaAwareValidator

try:
    from llm_sdk.llm_sdk import Small_LLM_Model
except ModuleNotFoundError as e:
    print(
        "Unable to import llm_sdk. Make sure you have copied the "
        "'llm_sdk' folder into the same directory containing 'src'. "
        f"Details: {e}"
    )
    sys.exit(1)


def get_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling in LLMs")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition file",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input file",
    )
    parser.add_argument(
        "--output",
        default="data/output",
        help="Path of the function to be return"
    )
    return parser.parse_args()


def get_json_from_file(name: str) -> Any:
    try:
        with open(name, "r", encoding="utf-8") as f:
            res_json = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: {e}") from e
    except IsADirectoryError as e:
        raise IsADirectoryError(
            f"Path is a directory: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error: {e}") from e
    return res_json


def get_line_from_file(name: str) -> dict[tuple[str, str], int]:
    try:
        with open(name, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied: {e}") from e
    except IsADirectoryError as e:
        raise IsADirectoryError(
            f"Path is a directory: {e}") from e
    except Exception as e:
        raise Exception(f"Error: {e}") from e
    res = {}
    for rank, line in enumerate(lines[1:]):
        part = line.split()
        res[(part[0], part[1])] = rank
    return res


def _predict_function_name(
    input_ids: list[int],
    model: Any,
    tokenizer: CustomTokenizer,
    schema: SchemaAwareValidator,
) -> str:
    current_name = ""

    while not schema.is_complete(current_name):
        forced = schema.get_unique_completion_for_name(current_name)
        if forced:
            current_name += forced
            input_ids.extend(tokenizer.encode(forced))
            break

        allowed_chars = schema.next_char_for_name(current_name)
        if schema.is_complete(current_name):
            allowed_chars.add('"')

        logits = model.get_logits_from_input_ids(input_ids)

        best_id = None
        best_score = float("-inf")

        for char in allowed_chars:
            candidate_ids = (
                tokenizer.get_tokens_starting_with(char)
                if hasattr(tokenizer, "get_tokens_starting_with")
                else []
            )
            if not candidate_ids:
                decoded_dict = getattr(tokenizer, "decoded_tokens", {})
                candidate_ids = [
                    tid
                    for tid, s in decoded_dict.items()
                    if s.startswith(char)
                ]

            for tid in candidate_ids:
                tok_str = tokenizer.decode(tid)
                trial = current_name + tok_str
                node = schema.name_trie
                valid_token = True

                for c in trial:
                    if c not in node:
                        valid_token = False
                        break
                    node = node[c]

                if valid_token and logits[tid] > best_score:
                    best_score = logits[tid]
                    best_id = tid

        if best_id is None:
            break

        chosen_str = tokenizer.decode(best_id)
        current_name += chosen_str
        input_ids.append(best_id)

    return current_name.strip('"')


def _extract_parameter_value(
    input_ids: list[int],
    param_type: str,
    model: Any,
    tokenizer: CustomTokenizer
) -> Any:
    value_str = ""
    max_steps = 60

    DIGITS_AND_SIGNS = set("0123456789.-")

    for _ in range(max_steps):
        logits = model.get_logits_from_input_ids(input_ids)
        best_id = None
        best_score = float("-inf")

        if param_type == "number":
            for tid in range(len(logits)):
                tok_str = tokenizer.decode(tid)

                if tok_str in (",", "}", " ", "\n") and len(value_str) > 0:
                    if logits[tid] > best_score:
                        best_score = logits[tid]
                        best_id = tid

                elif (all(c in DIGITS_AND_SIGNS for c in tok_str) and
                        len(tok_str) > 0):
                    if logits[tid] > best_score:
                        best_score = logits[tid]
                        best_id = tid

            if best_id is None:
                break

            chosen = tokenizer.decode(best_id)
            if chosen in (",", "}", " ", "\n"):
                break
            value_str += chosen
            input_ids.append(best_id)

        elif param_type == "string":

            for tid in range(len(logits)):
                tok_str = tokenizer.decode(tid)
                if '"' in tok_str:
                    if logits[tid] > best_score:
                        best_score = logits[tid]
                        best_id = tid
                else:
                    if logits[tid] > best_score:
                        best_score = logits[tid]
                        best_id = tid

            if best_id is None:
                break

            chosen = tokenizer.decode(best_id)
            if '"' in chosen:
                value_str += chosen.split('"')[0]
                break
            value_str += chosen
            input_ids.append(best_id)
        else:
            break

    if param_type == "number":
        try:
            return float(value_str) if "." in value_str else int(value_str)
        except ValueError:
            return 0.0
    return value_str


def generate_function_call(
    prompt: str,
    functions_txt: str,
    model: Any,
    tokenizer: CustomTokenizer,
    schema: SchemaAwareValidator,
) -> dict:
    prompt_txt = build_prompt(prompt, functions_txt)
    input_ids = tokenizer.encode(prompt_txt)

    start_prefix = '{\n  "name": "'
    input_ids.extend(tokenizer.encode(start_prefix))

    func_name = _predict_function_name(input_ids, model, tokenizer, schema)

    params_prefix = '",\n  "parameters": {'
    input_ids.extend(tokenizer.encode(params_prefix))

    required_params = schema.get_required_params(func_name)
    extracted_params: dict[str, Any] = {}

    for i, p_name in enumerate(required_params):
        p_type = schema.get_param_type(func_name, p_name)

        p_prefix = f'\n    "{p_name}": ' if i == 0 else f',\n    "{p_name}": '
        if p_type == "string":
            p_prefix += '"'
        input_ids.extend(tokenizer.encode(p_prefix))

        val = _extract_parameter_value(input_ids, p_type, model, tokenizer)
        extracted_params[p_name] = val

        if p_type == "string":
            input_ids.extend(tokenizer.encode('"'))

    input_ids.extend(tokenizer.encode("\n  }\n}"))

    return {"name": func_name, "parameters": extracted_params}


def main() -> None:
    argv = get_argv()
    model = Small_LLM_Model()
    try:
        merge_path = model.get_path_to_merges_file()
        tokenizer_path = model.get_path_to_tokenizer_file()
        vocab_path = model.get_path_to_vocab_file()
    except Exception as e:
        print(f"Error in method of Small_LLM_Model: {e}")
        sys.exit(1)
    try:
        dict_merge = get_line_from_file(merge_path)
        json_tokenizer = get_json_from_file(tokenizer_path)
        json_vocab = get_json_from_file(vocab_path)
        schema = SchemaAwareValidator(
            json_functions=get_json_from_file(argv.functions_definition)
        )
        json_prompts = get_json_from_file(argv.input)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    tokenizer = CustomTokenizer(
        token_id=json_vocab,
        dict_merge=dict_merge,
        special_token_id=json_tokenizer["added_tokens"],
    )
    function_txt = json.dumps(schema.json_functions, indent=2)
    results = []
    for p in json_prompts:
        prompt_text = p.get("prompt", "") if isinstance(p, dict) else p
        call_result = generate_function_call(
            prompt_text, function_txt, model, tokenizer, schema
        )
        final_obj = {
            "prompt": prompt_text,
            "name": call_result.get("name", "error_missing_name"),
            "parameters": call_result.get("parameters", {}),
        }

        if "error" in call_result:
            final_obj["error"] = call_result["error"]
            final_obj["raw_text"] = call_result.get("raw_text", "")

        results.append(final_obj)

    out_path = argv.output
    if os.path.isdir(out_path) or not out_path.endswith(".json"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "function_calling_results.json")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"KeyboardInterrupt: {e}")
    except EOFError as e:
        print(f"EOFError: {e}")
