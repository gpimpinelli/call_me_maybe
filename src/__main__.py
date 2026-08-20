from typing import Any
import sys
import json
import argparse
import os
import time

# Import dei moduli del nostro pacchetto: se uno di questi fallisce e' un
# errore di progetto (file mancante/rinominato), quindi lasciamo che
# l'eccezione si propaghi normalmente con il suo traceback originale,
# invece di nasconderla dietro un except generico.
from .custom_tokenizer import CustomTokenizer
from .json_validator import JsonValidator, State
from .utils import soft_max, get_logit_max, build_prompt
from .schema_aware_validator import SchemaAwareValidator
from .schema_constrained_decoder import SchemaConstrainedDecoder

# llm_sdk invece va copiato manualmente accanto a src/ (vedi subject IV.3.1):
# e' l'unico import per cui ha senso un messaggio d'errore dedicato e
# un'uscita pulita, perche' l'utente potrebbe semplicemente non averlo
# ancora copiato nella posizione corretta.
try:
    from llm_sdk.llm_sdk import Small_LLM_Model
except ModuleNotFoundError as e:
    print(
        "Impossibile importare llm_sdk. Assicurati di aver copiato la cartella "
        f"'llm_sdk' nella stessa directory che contiene 'src'. Dettagli: {e}"
    )
    sys.exit(1)

TOT_TIME = 0

def get_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling in LLMs")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json", help="Path to functions definition file")
    parser.add_argument("--input", default="data/input/function_calling_tests.json", help="Path to input file")
    parser.add_argument("--output", default="data/output", help="Path of the function to be return")
    return parser.parse_args()


def get_json_from_file(name: str) -> Any:
    try:
        with open(name, 'r', encoding="utf-8") as f:
            res_json = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}", file=sys.stderr) from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {e}", file=sys.stderr) from e
    except IsADirectoryError as e:
        raise IsADirectoryError(f"Path is a directory: {e}", file=sys.stderr) from e
    except Exception as e:
        raise Exception(f"Unexpected error: {e}, file=sys.stderr") from e
    return res_json


def get_line_from_file(name: str) -> dict[tuple[str, str], int]:
    try:
        with open(name, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}", file=sys.stderr) from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {e}", file=sys.stderr) from e
    except IsADirectoryError as e:
        raise IsADirectoryError(f"Path is a directory: {e}", file=sys.stderr) from e
    except Exception as e:
        raise Exception(f"Error: {e}") from e
    res = {}
    for rank, line in enumerate(lines[1:]):
        part = line.split()
        res[(part[0], part[1])] = rank
    return res


def get_valid_tokens(decoder: SchemaConstrainedDecoder, tokenizer: CustomTokenizer, allowed_cache: dict) -> list[int]:
    valid_ids = []

    val_text = decoder.validator.text
    val_status = decoder.validator.status
    val_stack_tup = tuple(decoder.validator.stack)
    val_key_path_tup = tuple(decoder.validator.current_key_path())

    def cached_allowed(text_so_far: str, status, json_stack_tup: tuple, key_stack_tup: tuple) -> set:
        # FASE 3: Estrazione precisa di partial e fix su chiavi di base
        needs_partial = (
            (status == State.IN_STRING and key_stack_tup and key_stack_tup[-1] == "name") or
            (status == State.IN_KEY_STRING and len(key_stack_tup) in (1, 2))
        )
        
        partial = None
        if needs_partial:
            full_text = val_text + text_so_far
            start = full_text.rfind('"')
            partial = full_text[start + 1:]
        
        stack_top = json_stack_tup[-1] if json_stack_tup else None
        key = (status, stack_top, key_stack_tup, partial)

        cached = allowed_cache.get(key)
        if cached is None:
            full_text_for_schema = val_text + text_so_far
            # FIX MEMORIA: Rimosso il frozenset che bloccava le performance
            cached = decoder.allowed_next_chars(
                text=full_text_for_schema, 
                status=status, 
                stack=list(json_stack_tup), 
                path=list(key_stack_tup)
            )
            allowed_cache[key] = cached
        return cached

    stack = [(
        tokenizer.trie_token,
        "",
        val_status,
        val_stack_tup,
        val_key_path_tup
    )]

    # --- COSTANTI FASE 4 ---
    END_TOKEN_CHARS = {']', '}', ',', ' ', '\n', '\t', '\r'}

    while stack:
        node, text_so_far, status, json_stack_tup, key_stack_tup = stack.pop()

        if None in node:
            valid_ids.extend(node[None])

        allowed = cached_allowed(text_so_far, status, json_stack_tup, key_stack_tup)

        for char, next_node in node.items():
            if char is None:
                continue

            if char in allowed:
                # --- FASE 4: MASSIMIZZAZIONE DEL FAST-PATH ---
                is_fast_path = False
                
                if status in (State.IN_STRING, State.IN_KEY_STRING) and char not in ('"', '\\'):
                    is_fast_path = True
                elif status in (State.IN_NUMBER, State.IN_SPECIAL) and char not in END_TOKEN_CHARS:
                    is_fast_path = True

                if is_fast_path:
                    stack.append((next_node, text_so_far + char, status, json_stack_tup, key_stack_tup))
                else:
                    new_text, new_status, new_stack, new_key_stack = decoder.advance_hypothetical(
                        text=val_text + text_so_far,
                        status=status,
                        stack=list(json_stack_tup),
                        path=list(key_stack_tup),
                        extra_char=char
                    )

                    if new_status != State.BAD:
                        stack.append((
                            next_node, 
                            text_so_far + char, 
                            new_status, 
                            tuple(new_stack), 
                            tuple(new_key_stack)
                        ))
    return valid_ids


def get_forced_completion(decoder: SchemaConstrainedDecoder) -> str | None:
    """Se lo schema determina in modo univoco il resto del nome funzione o del
    nome parametro corrente, ritorna il suffisso da forzare senza consultare
    il modello. None se serve comunque una scelta (o non siamo in uno di
    questi due casi).
    """
    path = decoder.validator.current_key_path()
    status = decoder.validator.status
    text = decoder.validator.text

    if path and path[-1] == "name" and status == State.IN_STRING:
        start = text.rfind('"')
        partial = text[start + 1:]
        return decoder.func_schema.get_unique_completion_for_name(partial)

    if len(path) == 1 and path[0] == "parameters" and status == State.IN_KEY_STRING:
        if decoder.resolved_function_name is None:
            return None
        start = text.rfind('"')
        partial = text[start + 1:]
        return decoder.func_schema.get_unique_completion_for_param(
            decoder.resolved_function_name, partial
        )

    return None


def generate_function_call(prompt: str, functions_txt: str, model, tokenizer, schema) -> dict:
    prompt_txt = build_prompt(prompt, functions_txt)
    input_ids = tokenizer.encode(prompt_txt)

    validator = JsonValidator(text="")
    decoder = SchemaConstrainedDecoder(validator=validator, func_schema=schema)
    allowed_cache: dict[tuple, set] = {}

    for step_i in range(300):
        if decoder.validator.is_complete():
            break

        forced = decoder.allowed_next_chars()
        if len(forced) == 1:
            forced_char = next(iter(forced))
            decoder.consume(forced_char)
            input_ids.extend(tokenizer.encode(forced_char))
            continue

        forced_completion = get_forced_completion(decoder)
        if forced_completion:
            decoder.consume(forced_completion)
            input_ids.extend(tokenizer.encode(forced_completion))
            continue

        logits = model.get_logits_from_input_ids(input_ids)

        valid_token_ids = get_valid_tokens(decoder, tokenizer, allowed_cache)

        if not valid_token_ids:
            break

        # --- FIX: TORNIAMO AL PURE PYTHON (È MOSTRUOSAMENTE PIÙ VELOCE QUI) ---
        if len(valid_token_ids) == 1:
            max_id = valid_token_ids[0]
        else:
            # Estrae l'ID con il valore logit più alto sfiorando solo i token validi
            # bypassando completamente la conversione costosa dei 150k token in numpy
            max_id = max(valid_token_ids, key=lambda vid: logits[vid])

        token_str = tokenizer.decode(max_id)
        decoder.consume(token_str)
        input_ids.append(max_id)

    try:
        result = json.loads(decoder.validator.text)
    except json.JSONDecodeError:
        result = {"error": "JSON malformato", "raw_text": decoder.validator.text}

    return result


def main():
    argv = get_argv()
    model = Small_LLM_Model()
    print(model._device)
    try:
        merge_path = model.get_path_to_merges_file()
        tokenizer_path = model.get_path_to_tokenizer_file()
        vocab_path = model.get_path_to_vocab_file()
    except Exception as e:
        print(f"Error in method of Small_LLM_Model: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        dict_merge = get_line_from_file(merge_path)
        json_tokenizer = get_json_from_file(tokenizer_path)
        json_vocab = get_json_from_file(vocab_path)
        schema = SchemaAwareValidator(json_functions=get_json_from_file(argv.functions_definition))
        json_prompts = get_json_from_file(argv.input)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    tokenizer = CustomTokenizer(token_id=json_vocab,dict_merge=dict_merge, special_token_id=json_tokenizer['added_tokens'])
    function_txt = json.dumps(schema.json_functions, indent=2)
    results = []
    for item in json_prompts:
        prompt_text = item.get("prompt", "") if isinstance(item, dict) else item
        start_time = time.time()
        call_result = generate_function_call(prompt_text, function_txt, model, tokenizer, schema)
        end_time = time.time()
        global TOT_TIME
        TOT_TIME += round(end_time - start_time, 2)

        final_obj = {
            "prompt": prompt_text,
            "name": call_result.get("name", "error_missing_name"),
            "parameters": call_result.get("parameters", {})
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
        
    print(f"finish in {TOT_TIME / 60} min")

   
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"KeyboardInterrupt: {e}", file=sys.stderr)
    except EOFError as e:
        print(f"EOFError: {e}", file=sys.stderr)