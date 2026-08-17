from typing import Any
import sys
import json
import argparse
import os

try:
    from .custom_tokenizer import CustomTokenizer
    from .json_validator import JsonValidator
    from llm_sdk.llm_sdk import Small_LLM_Model
    from .utils import soft_max, get_logit_max, build_prompt
    from .schema_aware_validator import SchemaAwareValidator
except Exception as e:
    print(f"Impossible load module llm_sdk: {e}")


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
    for p in json_prompts:
        llm_prompt = build_prompt(p['prompt'], function_txt)
        input_ids = tokenizer.encode(llm_prompt)
        json_validator = JsonValidator(text='')
        genetated_ids: list[int] = []
        MAX_TOKEN = 200
        while not json_validator.is_complete() and len(genetated_ids) < MAX_TOKEN:
        
    #print(schema.name_trie)
    #print(schema.function_by_name)
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"KeyboardInterrupt: {e}")
    except EOFError as e:
        print(f"EOFError: {e}")
#    except Exception as e:
#        print(f"Error: {e}", file=sys.stderr)