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
from .utils import build_prompt
from .schema_aware_validator import SchemaAwareValidator


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


def _predict_function_name(
    input_ids: list[int],
    model: Any,
    tokenizer: CustomTokenizer,
    schema: SchemaAwareValidator
) -> str:
    """Fa scegliere all'LLM il nome della funzione tramite constrained decoding sui logit."""
    current_name = ""
    
    while not schema.is_complete(current_name):
        # 1. Se lo schema ha un completamento univoco forzato, applicalo a costo 0
        forced = schema.get_unique_completion_for_name(current_name)
        if forced:
            current_name += forced
            input_ids.extend(tokenizer.encode(forced))
            break
            
        allowed_chars = schema.next_char_for_name(current_name)
        if schema.is_complete(current_name):
            allowed_chars.add('"')
            
        # 2. Ottieni logit dal modello
        logits = model.get_logits_from_input_ids(input_ids)
        
        # 3. Maschera i token: ammetti solo quelli che continuano prefissi validi
        best_id = None
        best_score = float("-inf")
        
        # Consideriamo solo i token che iniziano con uno dei caratteri permessi
        for char in allowed_chars:
            candidate_ids = tokenizer.get_tokens_starting_with(char) if hasattr(tokenizer, "get_tokens_starting_with") else []
            # Fallback se usi ancora la lista/dizionario base
            if not candidate_ids:
                candidate_ids = [tid for tid, s in getattr(tokenizer, "decoded_tokens", {}).items() if s.startswith(char)]
                
            for tid in candidate_ids:
                tok_str = tokenizer.decode(tid)
                # Verifica se l'intero token è un prefisso valido nel trie delle funzioni
                trial = current_name + tok_str
                # Controllo di coerenza con il trie dei nomi
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
    """Fa estrarre all'LLM il valore del parametro vincolandolo al tipo (number o string) via DFA."""
    value_str = ""
    max_steps = 60  # limite di sicurezza per stringhe lunghe
    
    DIGITS_AND_SIGNS = set("0123456789.-")
    
    for _ in range(max_steps):
        logits = model.get_logits_from_input_ids(input_ids)
        best_id = None
        best_score = float("-inf")
        
        if param_type == "number":
            # DFA per i numeri: ammette solo cifre/segni o terminatore (virgola o graffa)
            for tid in range(len(logits)):
                tok_str = tokenizer.decode(tid)
                # Se è il carattere di chiusura del numero
                if tok_str in (",", "}", " ", "\n") and len(value_str) > 0:
                    if logits[tid] > best_score:
                        best_score = logits[tid]
                        best_id = tid
                # Se è una continuazione valida del numero
                elif all(c in DIGITS_AND_SIGNS for c in tok_str) and len(tok_str) > 0:
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
            # DFA per le stringhe: accetta testo finché il modello non emette il delimitatore "
            for tid in range(len(logits)):
                tok_str = tokenizer.decode(tid)
                if '"' in tok_str:
                    # Token di chiusura o che contiene virgolette finali
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
                # Prendi solo la parte prima della virgoletta di chiusura
                value_str += chosen.split('"')[0]
                break
            value_str += chosen
            input_ids.append(best_id)
        else:
            break

    # Cast corretto del dato
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
    schema: SchemaAwareValidator
) -> dict:
    """Esegue la generazione ibrida: sintassi deterministica + inferenza mirata."""
    prompt_txt = build_prompt(prompt, functions_txt)
    input_ids = tokenizer.encode(prompt_txt)
    
    # 1. Inietta l'apertura JSON (0 chiamate neuroni)
    start_prefix = '{\n  "name": "'
    input_ids.extend(tokenizer.encode(start_prefix))
    
    # 2. SELEZIONE FUNZIONE (Guidata esclusivamente dai logit del modello)
    func_name = _predict_function_name(input_ids, model, tokenizer, schema)
    
    # 3. Inietta il passaggio ai parametri (0 chiamate neuroni)
    params_prefix = '",\n  "parameters": {'
    input_ids.extend(tokenizer.encode(params_prefix))
    
    # 4. Recupera i parametri attesi dallo schema Pydantic
    required_params = schema.get_required_params(func_name)
    extracted_params: dict[str, Any] = {}
    
    # 5. ESTRAZIONE PARAMETRI (Inferenza mirata + DFA di tipo)
    for i, p_name in enumerate(required_params):
        p_type = schema.get_param_type(func_name, p_name)
        
        # Inietta la chiave del parametro deterministicamente
        p_prefix = f'\n    "{p_name}": ' if i == 0 else f',\n    "{p_name}": '
        if p_type == "string":
            p_prefix += '"'
        input_ids.extend(tokenizer.encode(p_prefix))
        
        # Estrai il valore con decodifica vincolata al tipo
        val = _extract_parameter_value(input_ids, p_type, model, tokenizer)
        extracted_params[p_name] = val
        
        # Inietta la chiusura della stringa se applicabile
        if p_type == "string":
            input_ids.extend(tokenizer.encode('"'))
            
    # 6. Inietta la chiusura dell'oggetto (0 chiamate neuroni)
    input_ids.extend(tokenizer.encode('\n  }\n}'))
    
    return {
        "name": func_name,
        "parameters": extracted_params
    }


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