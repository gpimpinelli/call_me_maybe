from typing import Any
import sys
import json
import argparse
import os
import numpy as np

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
        raise FileNotFoundError(f"File not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {e}") from e
    except IsADirectoryError as e:
        raise IsADirectoryError(f"Path is a directory: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error: {e}") from e
    return res_json

def get_line_from_file(name: str) -> dict[tuple[str, str], int]:
    try:
        with open(name, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {e}") from e
    except IsADirectoryError as e:
        raise IsADirectoryError(f"Path is a directory: {e}") from e
    except Exception as e:
        raise Exception(f"Error: {e}") from e
    res = {}
    for rank, line in enumerate(lines[1:]):
        part = line.split()
        res[(part[0], part[1])] = rank
    return res

def get_valid_tokens(decoder: SchemaConstrainedDecoder, tokenizer: CustomTokenizer) -> list[int]:
    valid_ids = []

    # Cache locale allo step corrente: molte diramazioni della DFS condividono lo
    # stesso (status, stack, path) e quindi lo stesso insieme di caratteri ammessi,
    # evitando di ripetere rfind()/ricerca nel trie dello schema ad ogni nodo.
    allowed_cache: dict[tuple, frozenset] = {}

    def cached_allowed(text_full: str, status, json_stack: list, key_stack: list) -> frozenset:
        needs_partial = (
            (key_stack and key_stack[-1] == "name" and status == State.IN_STRING)
            or (len(key_stack) == 1 and key_stack[0] == "parameters" and status == State.IN_KEY_STRING)
        )
        if needs_partial:
            start = text_full.rfind('"')
            key = (status, tuple(json_stack), tuple(key_stack), text_full[start + 1:])
        else:
            key = (status, tuple(json_stack), tuple(key_stack), None)

        cached = allowed_cache.get(key)
        if cached is None:
            cached = frozenset(decoder.allowed_next_chars(
                text=text_full, status=status, stack=json_stack, path=key_stack
            ))
            allowed_cache[key] = cached
        return cached

    # Lo stack della nostra DFS conterrà tuple di:
    # (nodo_del_trie, testo_del_token_finora, stato_json, stack_parentesi_json, stack_chiavi_json)
    stack = [(
        tokenizer.trie_token,
        "",
        decoder.validator.status,
        list(decoder.validator.stack),
        decoder.validator.current_key_path()
    )]

    while stack:
        node, text_so_far, status, json_stack, key_stack = stack.pop()

        # Se il nodo corrente ha la chiave None, significa che abbiamo trovato
        # la fine di una stringa che corrisponde a uno o più token validi.
        if None in node:
            valid_ids.extend(node[None])

        # Chiediamo (con cache) quali caratteri possiamo digitare ORA
        allowed = cached_allowed(decoder.validator.text + text_so_far, status, json_stack, key_stack)

        # Esploriamo i figli del nodo corrente nel Trie
        for char, next_node in node.items():
            if char is None:
                continue

            # Se il carattere figlio è fra quelli permessi...
            if char in allowed:
                # ...simuliamo cosa succederebbe al JSON aggiungendo quel carattere
                new_text, new_status, new_stack, new_key_stack = decoder.advance_hypothetical(
                    text=decoder.validator.text + text_so_far,
                    status=status,
                    stack=list(json_stack),
                    path=list(key_stack),
                    extra_char=char
                )

                # Se l'aggiunta di questo carattere non rompe il JSON (cioè lo stato non è BAD),
                # allora continuiamo l'esplorazione lungo questo ramo!
                if new_status != State.BAD:
                    stack.append((next_node, new_text, new_status, new_stack, new_key_stack))

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
    # 1. Costruiamo il prompt usando la funzione che hai in utils.py
    prompt_txt = build_prompt(prompt, functions_txt)
    print(f"{prompt_txt}")
    input_ids = tokenizer.encode(prompt_txt)

    # 2. Inizializziamo il validatore JSON e il nostro decoder vincolato
    validator = JsonValidator(text="")
    print(validator.text)
    decoder = SchemaConstrainedDecoder(validator=validator, func_schema=schema)

    # Loop di generazione (impostiamo un limite max di token per sicurezza)
    for step_i in range(300):
        # Se il JSON è strutturalmente completo, ci fermiamo!
        if decoder.validator.is_complete():
            break

        # --- Short-circuit 1: se in questo stato esiste un solo carattere ammesso,
        # non serve interrogare l'LLM ne' esplorare il trie: lo forziamo direttamente.
        # Copre i caratteri "strutturali" (virgole, due punti, parentesi di chiusura).
        forced = decoder.allowed_next_chars()
        if len(forced) == 1:
            forced_char = next(iter(forced))
            decoder.consume(forced_char)
            input_ids.extend(tokenizer.encode(forced_char))
            continue

        # --- Short-circuit 2: se il prefisso gia' scritto del nome funzione o del
        # nome parametro individua UNA SOLA possibile parola nello schema, forziamo
        # tutto il resto della stringa in un colpo solo, senza chiamare il modello
        # per ogni singolo carattere restante.
        forced_completion = get_forced_completion(decoder)
        if forced_completion:
            decoder.consume(forced_completion)
            input_ids.extend(tokenizer.encode(forced_completion))
            continue

        # 3. Otteniamo i logits dall'LLM per i token passati finora
        logits = model.get_logits_from_input_ids(input_ids)

        # 4. Otteniamo la lista di ID dei token permessi tramite la nostra DFS
        valid_token_ids = get_valid_tokens(decoder, tokenizer)

        if not valid_token_ids:
            break  # Vicolo cieco (non dovrebbe succedere con schema solidi)

        # 5. Creiamo una maschera: tutti i logits partono da -infinito
        masked_logits = np.full(len(logits), -np.inf)

        # 6. Ripristiniamo il logit originale SOLO per i token permessi
        for valid_id in valid_token_ids:
            masked_logits[valid_id] = logits[valid_id]

        # 7. Scegliamo il token col valore logit più alto (Greedy Decoding)
        max_id = get_logit_max(masked_logits)

        # 8. Decodifichiamo l'ID nella stringa corrispondente
        token_str = tokenizer.decode(max_id)

        # 9. Facciamo "ingoiare" il token al decoder per far avanzare lo stato del JSON
        decoder.consume(token_str)

        # 10. Aggiungiamo il token scelto alla "memoria" per il turno successivo dell'LLM
        input_ids.append(max_id)

    # Alla fine del loop, il testo processato dal validatore dovrebbe essere un JSON valido
    try:
        result = json.loads(decoder.validator.text)
    except json.JSONDecodeError:
        result = {"error": "JSON malformato", "raw_text": decoder.validator.text}

    return result


def main():
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
        schema = SchemaAwareValidator(json_functions=get_json_from_file(argv.functions_definition))
        json_prompts = get_json_from_file(argv.input)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    tokenizer = CustomTokenizer(token_id=json_vocab,dict_merge=dict_merge, special_token_id=json_tokenizer['added_tokens'])
    function_txt = json.dumps(schema.json_functions, indent=2)
    results = []
    print(f"Inizio elaborazione di {len(json_prompts)} prompt...")
    
    for item in json_prompts:
        # Prende la stringa dalla chiave 'prompt'
        # (potrebbe essere una stringa diretta o un dizionario a seconda del test JSON)
        prompt_text = item.get("prompt", "") if isinstance(item, dict) else item
        
        print(f"-> Generazione per: '{prompt_text}'")
        
        # Chiamata al modello per generare l'output vincolato
        call_result = generate_function_call(prompt_text, function_txt, model, tokenizer, schema)
        
        # Formattiamo l'oggetto risultante esattamente come richiesto dal subject:
        # 1. prompt
        # 2. name
        # 3. parameters
        final_obj = {
            "prompt": prompt_text,
            "name": call_result.get("name", "error_missing_name"),
            "parameters": call_result.get("parameters", {})
        }
        
        # Se qualcosa è andato storto nella decodifica JSON, teniamo traccia
        if "error" in call_result:
            final_obj["error"] = call_result["error"]
            final_obj["raw_text"] = call_result.get("raw_text", "")
            
        results.append(final_obj)
        
    # Gestione del percorso di output (crea le cartelle se non esistono)
    out_path = argv.output
    # Se l'utente ha passato una cartella invece di un file, aggiungiamo il nome file atteso
    if os.path.isdir(out_path) or not out_path.endswith(".json"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "function_calling_results.json")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Finito! Risultati salvati correttamente in: {out_path}")

   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"KeyboardInterrupt: {e}")
    except EOFError as e:
        print(f"EOFError: {e}")
#    except Exception as e:
#        print(f"Error: {e}", file=sys.stderr)