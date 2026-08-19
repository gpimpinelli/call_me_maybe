# 🗺️ Roadmap Completa — Call Me Maybe (con Bonus)

## 📋 Analisi del Progetto

### Obiettivo
Implementare un sistema di **function calling** su un LLM (Qwen3-0.6B) che:
1. Riceve prompt in linguaggio naturale
2. Usa **constrained decoding** token-by-token per garantire output JSON **100% valido**
3. Seleziona la funzione giusta e ne estrae gli argomenti tipizzati

### Stato Attuale — Cosa È Già Fatto ✅

| File | Stato | Note |
|------|-------|-------|
| `src/custom_tokenizer.py` | ✅ Completo | BPE encode/decode custom, trie dei token, gestione special token |
| `src/json_validator.py` | ✅ Completo | FSM a stati per validazione JSON incrementale, `next_char()`, `is_valid_continuation()` |
| `src/schema_aware_validator.py` | ⚠️ Parziale | Struttura base OK, `param_trie_for()` usa `self.param_trie` (bug: dovrebbe essere `_param_tries`) |
| `src/utils.py` | ✅ Completo | `soft_max`, `get_logit_max`, `build_prompt` |
| `src/__main__.py` | ❌ Incompleto | Carica tutto ma il loop di generazione è mancante — si ferma dopo `function_txt = ...` |
| `Makefile` | ✅ Completo | Tutte le regole obbligatorie presenti |
| `pyproject.toml` | ✅ Completo | Dipendenze corrette |
| `README.md` | ❌ Mancante | Non esiste ancora |
| `data/output/function_calling_results.json` | ❌ Errato | Output attuale è malformato (nomi funzione senza `_`, valori sbagliati, campi con `:`) |

### Problema Critico nell'Output Attuale
```json
// SBAGLIATO — come appare ora:
{"name": "fnaddnumbers", "parameters": {"a": 2, "b": 3}}  // underscore mancanti
{"name": ":fngetsquareroot", ":parameters": {"arg1": ":shrek"}}  // coloni errati

// CORRETTO — come deve essere:
{"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```
Il constrained decoding **non è ancora integrato** nel loop di generazione.

---

## 🎯 Regole Fondamentali del Progetto

| Regola | Dettaglio |
|--------|-----------|
| **Linguaggio** | Python 3.10+ |
| **Stile** | flake8 + mypy (no `--strict` obbligatorio, ma raccomandato) |
| **Pydantic** | Tutte le classi DEVONO usare pydantic |
| **Dipendenze permesse** | `numpy`, `json`, `pydantic` |
| **Dipendenze vietate** | `dspy`, `pytorch` (uso diretto), `transformers` (uso diretto), `outlines` |
| **Modello** | `Qwen/Qwen3-0.6B` (default obbligatorio) |
| **Accesso SDK** | Solo metodi pubblici: `get_logits_from_input_ids`, `get_path_to_vocab_file`, `encode`, `decode` (opz.) |
| **Selezione funzione** | Tramite LLM, NON con euristiche/regex/keyword matching |
| **Output JSON** | 100% valido, sempre parseable, conforme allo schema |
| **Comando run** | `uv run python -m src [--functions_definition ...] [--input ...] [--output ...]` |

---

## 🚀 Roadmap Dettagliata

---

### FASE 1 — Fix Bug Critici (⏱ ~30 min)

> **Priorità massima**: il progetto non funziona senza questi fix.

#### 1.1 Fix `schema_aware_validator.py`

**File:** [`src/schema_aware_validator.py`](file:///Ubuntu/home/gianl/python/call_me_maybe/src/schema_aware_validator.py)

**Bug:** `param_trie_for()` usa `self.param_trie` invece di `self._param_tries`

```python
# PRIMA (bug):
def param_trie_for(self, function_name: str) -> dict:
    return self.param_trie.get(function_name, {})

# DOPO (fix):
def param_trie_for(self, function_name: str) -> dict:
    return self._param_tries.get(function_name, {})
```

**Aggiungere anche** il metodo per sapere il tipo atteso di un parametro:
```python
def get_param_type(self, function_name: str, param_name: str) -> str:
    """Return the expected type of a parameter ('number', 'string', 'boolean')."""
    func = self._function_by_name.get(function_name, {})
    params = func.get("parameters", {})
    return params.get(param_name, {}).get("type", "string")

def get_required_params(self, function_name: str) -> list[str]:
    """Return list of all parameter names for a function."""
    func = self._function_by_name.get(function_name, {})
    return list(func.get("parameters", {}).keys())
```

#### 1.2 Fix `JsonValidator` — Integrazione con Schema

Il `JsonValidator` attuale valida la **struttura JSON generica** ma non conosce lo schema atteso (`name`, `parameters`, tipi). Serve estenderlo per:

- Quando la chiave corrente è `"name"` → il valore deve essere un nome funzione valido (usare trie di `SchemaAwareValidator`)
- Quando la chiave corrente è un parametro di tipo `"number"` → accettare solo token numerici
- Quando la chiave corrente è un parametro di tipo `"string"` → accettare solo stringhe

**Approccio:** creare un `SchemaConstrainedDecoder` che coordina `JsonValidator` + `SchemaAwareValidator`.

---

### FASE 2 — Core: Loop di Generazione con Constrained Decoding (⏱ ~3-4 ore)

> Questo è il **cuore del progetto** — quello che manca completamente.

#### 2.1 Architettura del Decoder

Creare il file [`src/constrained_decoder.py`](file:///Ubuntu/home/gianl/python/call_me_maybe/src/constrained_decoder.py)

**Logica del loop:**
```
prompt → tokenize → input_ids → loop:
    1. get_logits_from_input_ids(input_ids)
    2. per ogni token nel vocabolario:
       a. converti token_id → stringa
       b. verifica se quella stringa è un continuation valido di generated_so_far
       c. se NON valido → logit = -inf
    3. argmax sui logit filtrati → token selezionato
    4. aggiungi token a input_ids
    5. ripeti finché JSON completo (validator.is_complete() == True)
    6. parse JSON → restituisce dict
```

**Pseudocodice strutturato:**
```python
class ConstrainedDecoder(BaseModel):
    model: Small_LLM_Model      # SDK
    tokenizer: CustomTokenizer  # tokenizer custom
    schema: SchemaAwareValidator

    def generate(self, prompt: str) -> dict:
        input_ids = self.tokenizer.encode(prompt)
        generated_text = ""
        validator = JsonValidator(text="")
        
        while not validator.is_complete():
            logits = self.model.get_logits_from_input_ids(input_ids)
            
            # Constrained filtering
            valid_logits = [-inf] * len(logits)
            for token_id, logit in enumerate(logits):
                token_str = self.tokenizer.decode(token_id)
                if self._is_valid_next(token_str, validator, generated_text):
                    valid_logits[token_id] = logit
            
            next_token_id = argmax(valid_logits)
            next_token_str = self.tokenizer.decode(next_token_id)
            
            generated_text += next_token_str
            input_ids.append(next_token_id)
            
            validator.text = generated_text
        
        return json.loads(generated_text)
```

#### 2.2 Validazione Token per Token — Il Punto Critico

La funzione `_is_valid_next(token_str, validator, current_text)` deve:

1. **Prendere il token candidato** (stringa)
2. **Simulare l'aggiunta** al testo generato finora
3. **Verificare** che il risultato sia ancora un JSON valido in costruzione

**Usare `is_valid_continuation(token_str)`** già presente in `JsonValidator` — non è necessario reinventarlo!

Attenzione: un token può rappresentare **più caratteri** (es. `" world"` è un singolo token). Il validator deve accettarli tutti insieme.

#### 2.3 Vincolo Schema sulle Stringhe Speciali

La parte più delicata: quando il decoder sta generando il **valore del campo `name`**, deve essere ristretto ai soli nomi di funzione validi.

**Come rilevare il contesto:**
- Usare `validator.current_key_path()` che restituisce la lista di chiavi annidate
- Se `key_path[-1] == "name"` → siamo nel valore del nome funzione
- Se `key_path[-2] == "parameters"` e `key_path[-1] == "un_param"` → siamo in un valore parametro

**Come restringere:**
- Per il **nome funzione**: usare il trie di `SchemaAwareValidator.name_trie`
- Per **parametri number**: bloccare token che contengono lettere
- Per **parametri string**: lasciare libero il testo

#### 2.4 Completare `__main__.py`

```python
# Dopo la riga: function_txt = json.dumps(schema.json_functions, indent=2)
# AGGIUNGERE:

decoder = ConstrainedDecoder(model=model, tokenizer=tokenizer, schema=schema)
results = []

for item in json_prompts:
    prompt = item.get("prompt", "")
    if not prompt:
        continue
    
    full_prompt = build_prompt(prompt, function_txt)
    
    try:
        result = decoder.generate(full_prompt)
        result["prompt"] = prompt  # aggiunge il campo obbligatorio
        results.append(result)
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}", file=sys.stderr)

# Scrivi output
output_path = os.path.join(argv.output, "function_calling_results.json")
os.makedirs(argv.output, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results written to {output_path}")
```

---

### FASE 3 — Verifica Output e Test (⏱ ~1-2 ore)

#### 3.1 Test Manuale

Eseguire `uv run python -m src` e verificare che `data/output/function_calling_results.json` contenga:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  ...
]
```

**Checklist output:**
- [ ] Nomi funzione esatti (con `_`)
- [ ] Nomi parametri esatti (come da `functions_definition.json`)
- [ ] Tipi corretti (`number` → float, `string` → str)
- [ ] Campo `prompt` presente in ogni oggetto
- [ ] JSON valido (nessuna trailing comma, nessun commento)

#### 3.2 Edge Cases da Testare

- Prompt ambiguo (es. "hello" → quale funzione?)
- Numeri grandi (265, 345)
- Stringhe con caratteri speciali
- Funzioni con 3+ parametri (`fn_substitute_string_with_regex`)

#### 3.3 Accuracy Target

Il progetto richiede **≥90% accuracy** e **100% valid JSON**.
Con constrained decoding la validità JSON è garantita by design.
L'accuracy dipende dal prompt: migliorare `build_prompt` se i risultati sono sbagliati.

---

### FASE 4 — Lint e Type Checking (⏱ ~1 ora)

```bash
make lint
# flake8 . — nessun errore
# mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports
#         --disallow-untyped-defs --check-untyped-defs
```

**Problemi comuni da fixare:**
- Type hints mancanti su tutte le funzioni
- `Any` non dichiarato esplicitamente
- Attributi privati pydantic senza tipo esplicito

---

### FASE 5 — README.md (⏱ ~1-2 ore)

**File:** [`README.md`](file:///Ubuntu/home/gianl/python/call_me_maybe/README.md) — **OBBLIGATORIO, mancante!**

Il README deve contenere **esattamente** questi elementi (da subject):

```markdown
*This project has been created as part of the 42 curriculum by gipimpin.*

## Description
[obiettivo del progetto, panoramica function calling]

## Instructions
[installazione, compilazione, esecuzione]
uv sync
uv run python -m src

## Algorithm Explanation
[spiegazione constrained decoding passo per passo]

## Design Decisions
[scelte tecniche: perché BPE custom, perché FSM, perché trie]

## Performance Analysis
[accuracy ottenuta, tempo di esecuzione, affidabilità]

## Challenges Faced
[difficoltà incontrate e soluzioni]

## Testing Strategy
[come hai validato l'implementazione]

## Example Usage
[esempi concreti di esecuzione]

## Resources
[riferimenti tecnici + descrizione uso AI]
```

---

## 🎁 FASE 6 — Bonus (⏱ ~4-8 ore)

> I bonus **devono essere funzionanti** — non solo descritti nel README.

### BONUS 1 — Supporto Multi-Modello ⭐

**Obiettivo:** Permettere modelli diversi da `Qwen/Qwen3-0.6B` via CLI.

**Implementazione:**
```bash
uv run python -m src --model "microsoft/Phi-3-mini-4k-instruct"
```

```python
# In __main__.py, aggiungere argomento:
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")

# In Small_LLM_Model (non modificabile), l'SDK già accetta model_name:
model = Small_LLM_Model(model_name=argv.model)
```

**File da creare:** `src/model_config.py` — pydantic model per configurazione modello.

### BONUS 2 — Tokenizer Ricopiato (NO encode/decode diretti) ⭐⭐

**Obiettivo:** Il bonus richiede di usare **solo** `get_logits_from_input_ids` e `get_path_to_vocab_file` — senza chiamare `model.encode()` e `model.decode()` nel codice principale.

**Già parzialmente fatto:** `CustomTokenizer` implementa BPE da zero.

**Cosa manca:**
- `__main__.py` non deve usare `model.encode()` / `model.decode()` 
- Usare esclusivamente `tokenizer.encode()` e `tokenizer.decode()` custom
- Aggiungere metodo `decode(ids: list[int]) -> str` al CustomTokenizer (alias di `decode_list`)

**Bonus extra richiesto:**
> "Public implementation of tokenizer encode and optional decode methods"
> "Demonstration of how encoding and decoding integrate with constrained decoding"

Aggiungere a `CustomTokenizer`:
```python
@property  
def vocab_size(self) -> int:
    return len(self.token_id)

def get_all_token_strings(self) -> dict[int, str]:
    """Return mapping id -> decoded string for ALL tokens."""
    result = {}
    for id_, token_str in self._id_token.items():
        try:
            result[id_] = self.decode(id_)
        except Exception:
            result[id_] = ""
    return result
```

### BONUS 3 — Error Recovery Avanzato ⭐

**Obiettivo:** Se il decoder si blocca (nessun token valido), recuperare.

**Implementazione:**
```python
# In ConstrainedDecoder.generate():
valid_count = sum(1 for v in valid_logits if v != -inf)

if valid_count == 0:
    # Strategia 1: backtrack di N token
    # Strategia 2: inserire token "sicuro" (es. chiudi la stringa corrente)
    # Strategia 3: terminare anticipatamente con JSON parziale completato
    raise ConstrainedDecodingError(f"No valid tokens at position {len(generated_text)}")
```

Creare eccezione custom `ConstrainedDecodingError` (classe pydantic o dataclass).

### BONUS 4 — Caching e Batching ⭐

**Obiettivo:** Ottimizzare le prestazioni.

**Caching:**
```python
# Cache dei logit per prefissi identici
from functools import lru_cache

@lru_cache(maxsize=128)
def _get_logits_cached(self, input_ids_tuple: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(self.model.get_logits_from_input_ids(list(input_ids_tuple)))
```

**Pre-computazione token validi:**
- Pre-filtrare il vocabolario all'avvio per ogni stato JSON possibile
- Memorizzare per ogni stato → set di token ID sempre validi

### BONUS 5 — Test Suite Completa ⭐

**File da creare:** `tests/` directory

```
tests/
├── test_tokenizer.py
├── test_json_validator.py
├── test_schema_validator.py
├── test_constrained_decoder.py
└── test_integration.py
```

**Test minimi per ciascun modulo:**

```python
# test_tokenizer.py
def test_encode_decode_roundtrip():
    text = "Hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode_list(ids)
    assert decoded == text

def test_special_tokens():
    ids = tokenizer.encode("<|im_start|>user")
    assert tokenizer.special_token_id["<|im_start|>"] in ids

# test_json_validator.py
def test_valid_json_complete():
    v = JsonValidator(text='{"a": 1}')
    assert v.is_complete() == True

def test_invalid_json():
    v = JsonValidator(text='{"a":')
    assert v.validate() != State.END

def test_next_char_start():
    v = JsonValidator(text="")
    assert "{" in v.next_char()

# test_constrained_decoder.py
def test_always_valid_json(decoder, prompts):
    for prompt in prompts:
        result = decoder.generate(prompt)
        json.loads(json.dumps(result))  # must not raise
```

**Eseguire con:**
```bash
uv run pytest tests/ -v
```

### BONUS 6 — Visualizzazione del Processo ⭐

**Obiettivo:** Mostrare token per token come il decoder sceglie il prossimo token.

**Implementazione:**
```python
class ConstrainedDecoder(BaseModel):
    verbose: bool = False

    def generate(self, prompt: str) -> dict:
        ...
        while not validator.is_complete():
            ...
            if self.verbose:
                top5 = sorted(enumerate(valid_logits), key=lambda x: -x[1])[:5]
                print(f"\n[Step {step}] Generated: {generated_text!r}")
                print(f"  Valid tokens: {valid_count}/{len(logits)}")
                for tid, logit in top5:
                    tok = self.tokenizer.decode(tid)
                    print(f"  → {tok!r:20s} (logit={logit:.2f})")
```

**Aggiungere flag CLI:**
```bash
uv run python -m src --verbose
```

### BONUS 7 — Supporto Argomenti Annidati ⭐⭐

**Obiettivo:** Gestire parametri di tipo `object` o `array` nella definizione funzione.

**Estendere `SchemaAwareValidator`:**
```python
# functions_definition.json esempio con nested:
{
  "name": "fn_create_user",
  "parameters": {
    "user": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
      }
    }
  }
}
```

Il `JsonValidator` già supporta oggetti annidati. Serve coordinarlo con il tipo schema per parametri di tipo `object`.

---

## 📊 Riepilogo Priorità

| Fase | Contenuto | Priorità | Tempo stimato |
|------|-----------|----------|---------------|
| 1 | Fix bug critici (param_trie, schema methods) | 🔴 CRITICA | 30 min |
| 2 | Loop generazione + constrained decoding | 🔴 CRITICA | 3-4 ore |
| 3 | Test manuale output + edge cases | 🔴 CRITICA | 1-2 ore |
| 4 | Lint + mypy clean | 🟡 ALTA | 1 ora |
| 5 | README.md completo | 🟡 ALTA | 1-2 ore |
| 6.1 | Bonus: multi-modello | 🟢 BONUS | 1 ora |
| 6.2 | Bonus: tokenizer ricreato (no encode/decode SDK) | 🟢 BONUS | 1-2 ore |
| 6.3 | Bonus: error recovery | 🟢 BONUS | 1 ora |
| 6.4 | Bonus: caching/batching | 🟢 BONUS | 1-2 ore |
| 6.5 | Bonus: test suite pytest | 🟢 BONUS | 2-3 ore |
| 6.6 | Bonus: visualizzazione | 🟢 BONUS | 1 ora |
| 6.7 | Bonus: argomenti annidati | 🟢 BONUS | 2-3 ore |

**Totale mandatory:** ~7-9 ore  
**Totale con tutti i bonus:** ~16-22 ore

---

## ⚠️ Vincoli da Rispettare Durante l'Implementazione

> [!IMPORTANT]
> - **NON** usare `model.encode()` o `model.decode()` nel codice main — usare solo `CustomTokenizer` (requisito bonus, ma anche buona pratica)
> - **NON** selezionare la funzione con regex o keyword matching — il modello deve farlo
> - **NON** includere `data/output/` nel repository Git
> - La **funzione chiamata deve essere scelta dal LLM** — il constrained decoding guida, ma non hardcoda la scelta

> [!WARNING]  
> Il `_STRING_CHARS` nel `JsonValidator` attualmente include TUTTI i codepoint Unicode tranne `"` e `\`. Per grandi vocabolari questo può essere lento — considerare di pre-filtrare i token accettabili.

> [!CAUTION]
> Il vocabolario di Qwen3-0.6B ha ~150,000 token. Iterare su tutti ad ogni passo è O(vocab_size) per token. Con prompt di 50 token e vocabolario 150k, sono ~7.5M operazioni per prompt. **Ottimizzare** con pre-filtering per stati.

---

## 🔍 File Critici da Completare

| File | Azione richiesta |
|------|-----------------|
| [`src/__main__.py`](file:///Ubuntu/home/gianl/python/call_me_maybe/src/__main__.py) | Aggiungere loop generazione dopo `function_txt = ...` |
| [`src/schema_aware_validator.py`](file:///Ubuntu/home/gianl/python/call_me_maybe/src/schema_aware_validator.py) | Fix `param_trie` → `_param_tries` + aggiungere `get_param_type()`, `get_required_params()` |
| [`src/constrained_decoder.py`](file:///Ubuntu/home/gianl/python/call_me_maybe/src/constrained_decoder.py) | **CREARE** — logica core del constrained decoding |
| [`README.md`](file:///Ubuntu/home/gianl/python/call_me_maybe/README.md) | **CREARE** — obbligatorio, attualmente mancante |
