Ecco la roadmap rivista, più leggibile e senza codice (tranne i file di configurazione iniziali). Ho mantenuto la struttura a fasi e ho spiegato ogni passaggio in modo chiaro.

---

# 🛠️ Roadmap Definitiva – Function Calling con Constrained Decoding

## Fase 0 – Setup e vincoli

- **Ambiente virtuale e dipendenze**
  - Inizializza il progetto con `uv init --app`.
  - Aggiungi `pydantic` e `numpy` con `uv add pydantic numpy`.
  - Copia la cartella `llm_sdk/` nella root del progetto. Non modificare i file interni.

- **Struttura delle directory**
  ```
  progetto/
  ├── pyproject.toml
  ├── uv.lock
  ├── Makefile
  ├── .gitignore
  ├── README.md
  ├── llm_sdk/                # copiato dall'originale
  ├── src/
  │   ├── __init__.py
  │   ├── __main__.py
  │   ├── models.py
  │   ├── decoder.py
  │   ├── grammar.py
  │   ├── tokenizer_utils.py
  │   └── utils.py
  └── data/
      └── input/
          ├── functions_definition.json
          └── function_calling_tests.json
  ```

- **File di configurazione obbligatori**

  **Makefile** (esattamente come richiesto):
  ```makefile
  .PHONY: install run debug clean lint lint-strict

  install:
  	uv sync

  run:
  	uv run python -m src

  debug:
  	uv run python -m pdb -m src

  clean:
  	rm -rf __pycache__ .mypy_cache data/output/

  lint:
  	flake8 .
  	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

  lint-strict:
  	flake8 .
  	mypy . --strict
  ```

  **pyproject.toml** (minimale, dopo aver aggiunto le dipendenze):
  ```toml
  [project]
  name = "call-me-maybe"
  version = "0.1.0"
  description = "Function calling with constrained decoding"
  requires-python = ">=3.10"
  dependencies = [
      "numpy>=1.24",
      "pydantic>=2.0",
  ]
  ```

  **.gitignore**:
  ```
  __pycache__/
  .mypy_cache/
  .venv/
  data/output/
  ```

- **Verifica iniziale**
  - Crea un `__main__.py` temporaneo che importa `Small_LLM_Model` per accertarti che il percorso funzioni.

---

## Fase 1 – Esplorazione del modello e del vocabolario

- **Carica il modello**  
  Crea un’istanza di `Small_LLM_Model()` per avere accesso ai metodi dell’SDK.

- **Vocabolario**  
  - Chiama `get_path_to_vocab_file()` e carica il JSON.  
  - Costruisci due mappe: `token → id` e `id → token`.  
  - Analizza la dimensione e l’aspetto dei token (molti hanno uno spazio iniziale rappresentato da `Ġ`).  
  - Annota i token speciali (fine testo, padding, ecc.).

- **Test di generazione libera**  
  - Tokenizza un prompt di esempio con `encode()`.  
  - Per un certo numero di passi, chiama `get_logits_from_input_ids()` e prendi il token con logit massimo (greedy).  
  - Decodifica i token per vedere il testo generato.  
  - Prova a far generare un JSON con un prompt: osserva come il modello spesso sbagli virgolette, virgole, parentesi. Questo conferma la necessità del constrained decoding.

- **Obiettivo di questa fase**  
  Avere familiarità con i metodi dell’SDK, la struttura dei logits e il formato dei token.

---

## Fase 2 – Motore di Constrained Decoding (il cuore del progetto)

L’obiettivo è costruire un sistema che genera un token alla volta, ma ad ogni passo filtra i logits per accettare solo i token che mantengono la stringa prodotta come prefisso di un JSON valido e conforme allo schema delle funzioni.

### 2.1 Approccio generale: automa a stati finiti (FSM) e vocabolario

- **FSM**  
  Una macchina a stati che rappresenta la grammatica JSON + le restrizioni dello schema. Tiene traccia dello stato attuale (es. “mi aspetto una chiave”, “dentro un numero”, “valore booleano”, ecc.) e per ogni carattere decide se la transizione è ammessa.

- **Trie dei token**  
  Poiché il modello produce token interi (non singoli caratteri), dobbiamo verificare che l’intera stringa del token sia compatibile con la FSM.  
  Un Trie (albero dei prefissi) organizzato sulle stringhe dei token permette di:
  - Navigare i caratteri iniziali consentiti dallo stato corrente.
  - Raccogliere solo i token che, carattere per carattere, rimangono validi.

- **Mascheramento dei logits**  
  - Si ottengono i logits dal modello.
  - Si determina l’insieme dei token consentiti interrogando il Trie e simulando la FSM.
  - Si impostano i logits di tutti gli altri token a `-inf` (usando numpy).
  - Si seleziona il token con logit massimo (o campionamento in bonus).

### 2.2 Stati principali della FSM (prima versione senza annidamenti)

- `START` → accetta solo token che iniziano con `{`.
- `OBJECT_OPEN` → dopo `{`, si aspetta `"` (inizio chiave `"function"`).
- `IN_KEY` → si sta generando una chiave; la stringa accumulata deve essere prefisso di una delle chiavi attese (es. `"function"`, poi `"arguments"`, poi le chiavi dei parametri). Quando la chiave è completa, si accetta il token che contiene la virgoletta di chiusura e eventuali altri caratteri (ad esempio `":`).
- `AFTER_COLON` → dopo i due punti, si aspetta l’inizio del valore. In base al tipo atteso, si transita nello stato appropriato:
  - `IN_STRING`
  - `IN_NUMBER`
  - `IN_BOOLEAN`
- `IN_STRING` → dopo la virgoletta di apertura, accetta tutti i caratteri che non contengono `"` non escapata (per semplicità vietiamo anche `\`). Quando compare un token che contiene `"`, la FSM valuta se dopo il `"` c’è un delimitatore valido (`,` o `}`) e transita di conseguenza.
- `IN_NUMBER` → segue la grammatica JSON dei numeri. Accetta token che formano un prefisso di numero valido (cifre, segno meno, punto decimale, esponente).
- `IN_BOOLEAN` → deve completare esattamente `true` o `false`. La FSM confronta il token con i prefissi di `true`/`false` e, una volta completata la parola, controlla che il carattere successivo sia un delimitatore.
- `COMMA_OR_CLOSE` → dopo un valore, se ci sono altri parametri attesi, aspetta `,`; altrimenti aspetta `}`. Dopo la parentesi chiusa finale, lo stato diventa `END` e si ammette solo il token di fine sequenza.

### 2.3 Integrazione dello schema

- Carica le definizioni delle funzioni dal file JSON.
- L’FSM conosce tutti i nomi funzione validi.
- Durante la generazione del nome funzione (dopo `"function": "`), l’FSM limita i token a quelli che sono prefissi di almeno uno dei nomi.
- Non appena viene completato un nome esatto, l’FSM fissa la funzione scelta, e da quel momento gli stati successivi riflettono i parametri di quella funzione (nomi, tipi, ordine).

**Scelta dell’ordine dei parametri**  
Per semplificare la FSM, puoi forzare un ordine fisso (ad esempio alfabetico). In questo modo sai già quali chiavi aspettarti e in quale sequenza. Il modello genererà solo i valori. Questo è accettabile e non viola i requisiti, perché la funzione e i valori sono comunque decisi dal modello. L’FSM si assicura che la sintassi sia corretta.

### 2.4 Algoritmo di mascheramento efficiente

- Invece di iterare su tutto il vocabolario, usa il Trie per ottenere solo i token che iniziano con i caratteri consentiti dallo stato corrente.
- Per ciascun token candidato, simula l’avanzamento della FSM carattere per carattere (con una copia dello stato). Se tutti i caratteri sono validi e lo stato finale è accettante, il token è valido.
- Ottimizzazione: per stati come `IN_STRING` (dove la maggior parte dei token è ammessa tranne quelli con `"`), si può pre‑filtrare controllando semplicemente la presenza di `"` nel token.

### 2.5 Loop di generazione principale

- Parti dal prompt tokenizzato.
- Inizializza l’FSM con l’elenco delle funzioni.
- Finché non raggiungi lo stato `END` o un numero massimo di token:
  - Ottieni i logits.
  - Ricava la maschera dei token validi.
  - Scegli il token con logit più alto.
  - Aggiorna la sequenza di input.
  - Aggiorna l’FSM con la stringa del token.
- Restituisci la stringa JSON completa.

---

## Fase 3 – Modelli Pydantic e validazione dell’output

- **Definisci le classi Pydantic** in `models.py`:
  - `FunctionDef`: con `name`, `description`, `parameters` (dizionario nome→tipo), `returns`.
  - `FunctionCallResult`: con `prompt`, `name`, `parameters`.
- **Carica e valida** i file di input usando questi modelli. In caso di errori, fornisci messaggi chiari e termina senza crash.
- **Dopo la generazione** del JSON, convertilo in oggetto Pydantic per garantire che i tipi corrispondano. Se necessario, applica conversioni (es. un numero scritto come intero può rimanere `float` se il tipo è `number` – come da esempi).
- **Scrivi l’output** in `data/output/function_calling_results.json` (o nel percorso specificato) come array di oggetti, senza chiavi extra.

---

## Fase 4 – Interfaccia da riga di comando

- In `src/__main__.py` usa `argparse` per accettare i tre argomenti opzionali.
- Default:
  - `--functions_definition`: `data/input/functions_definition.json`
  - `--input`: `data/input/function_calling_tests.json`
  - `--output`: `data/output/function_calling_results.json`
- Per ogni prompt letto dal file di input, chiama la funzione di generazione e raccogli i risultati.
- Gestisci sin da subito i casi di errore: file non trovato, JSON malformato, directory di output inesistente (creala).

---

## Fase 5 – Robustezza e gestione errori

- **Mai crash improvvisi**: tutte le operazioni di I/O e la generazione vanno protette con `try`/`except` specifici.
- **Risorse**: usa sempre `with` per file e context manager ove possibile.
- **Errori di generazione**: se la maschera diventa vuota (nessun token valido), solleva un’eccezione custom e nel main catturala producendo un messaggio di errore per quel prompt, ma proseguendo con i successivi.
- **Messaggi utente**: ogni errore deve essere riportato in modo chiaro sullo standard error.
- **Type hints e docstring**: ogni funzione deve avere annotazioni di tipo e una docstring in stile Google o NumPy.

---

## Fase 6 – Testing e validazione

- **Test con i prompt di esempio** forniti nel file `function_calling_tests.json`. Verifica:
  - Il JSON prodotto è sempre valido (prova a parsarlo).
  - Il nome della funzione e i parametri corrispondono a quelli attesi.
  - I tipi sono corretti (numeri come `float`, stringhe senza virgolette esterne).
- **Casi limite da aggiungere manualmente**:
  - Numeri negativi, zero, notazione scientifica (se permessa).
  - Stringhe vuote `""`.
  - Prompt ambigui (es. “somma 2 e 3” con più funzioni disponibili).
  - Funzioni con parametri booleani (aggiungile ai tuoi test, anche se non presenti nell’esempio).
- **Test di robustezza**: cancella un file di input, fornisci un JSON malformato, lancia il programma senza la cartella `llm_sdk/` – deve mostrare un errore e uscire pulitamente.
- **Performance**: misura il tempo totale di esecuzione su tutti i prompt di esempio. Deve restare sotto i 5 minuti. Se troppo lento, ottimizza il mascheramento (es. caching delle maschere per stati comuni).

---

## Fase 7 – README.md completo

Il file deve includere tutte le sezioni richieste:

- *Prima riga in corsivo con i login 42.*
- **Description**: scopo del progetto e panoramica.
- **Instructions**: come installare (`uv sync`), eseguire il comando base, compilare (non applicabile).
- **Resources**: link alla documentazione di Pydantic, numpy, articoli sul constrained decoding, e una dichiarazione su come hai usato l’intelligenza artificiale (es. per generare idee sulla FSM, correggere bozze, ecc.).
- **Algorithm explanation**: descrivi il tuo approccio: la FSM, il trie, il mascheramento dei logits, la scelta della funzione guidata dal modello.
- **Design decisions**: spiega perché hai scelto un ordine fisso per i parametri, come gestisci i numeri, perché il trie migliora le prestazioni, ecc.
- **Performance analysis**: riporta accuratezza (>90%), tempi di esecuzione, consumo di memoria.
- **Challenges faced**: difficoltà incontrate (es. gestione dei token con spazi, riconoscimento fine stringa, parsing dei numeri) e soluzioni adottate.
- **Testing strategy**: descrivi i test automatici e manuali effettuati.
- **Example usage**: mostra alcuni comandi e l’output atteso.

---

## Fase 8 – Bonus (opzionali, implementabili separatamente)

### B.1 Supporto multi‑modello
- Aggiungi un argomento `--model` al CLI. Istanzia `Small_LLM_Model` con quel nome. Verifica che il vocabolario e il tokenizer funzionino con altri modelli (es. altri Qwen). La logica FSM rimane invariata.

### B.2 Tokenizer proprio (evitare `encode`/`decode` dell’SDK)
- Usa `get_path_to_vocab_file()` e `get_path_to_merges_file()` per ottenere i dati.
- Implementa una tua funzione `encode(text)` che applica l’algoritmo BPE utilizzando le regole di merge.
- Implementa `decode(ids)` ricostruendo il testo dai token.
- Nel loop principale, tokenizza il prompt con la tua funzione e usa solo `get_logits_from_input_ids`.
- Questo dimostra piena comprensione del tokenizer.

### B.3 Error recovery avanzato
- Se la generazione si blocca (maschera vuota), riprova con una piccola temperatura (aggiunta di rumore gaussiano ai logits) oppure reintroducendo un piccolo numero di token scelti diversamente (backtracking).
- Oppure, se il JSON generato non è parsabile (non dovrebbe accadere), tenta un parsing “rilassato” con `json.JSONDecoder`.

### B.4 Ottimizzazioni
- **Caching**: per ogni stato FSM, salva l’insieme dei token validi (o la maschera) in modo da non ricalcolarlo se lo stato si ripresenta.
- **Vettorizzazione**: usa numpy per applicare la maschera in modo efficiente senza cicli Python su tutti i token.
- **Trie ottimizzato**: utilizza array di indici per i figli anziché dizionari, se il vocabolario è molto grande.

### B.5 Suite di test completa
- Scrivi test unitari con `pytest` per:
  - Trie: costruzione e ricerca.
  - FSM: transizioni valide e non valide per vari input.
  - Funzioni di mascheramento: verifica che i token proibiti vengano impostati a `-inf`.
  - Integrazione con un modello mock (che restituisce logits prevedibili) per testare l’intera pipeline.

### B.6 Visualizzazione del processo di generazione
- Aggiungi un’opzione `--verbose` che stampi ad ogni passo: stato FSM, token scelto, logit massimo, maschera applicata.
- In alternativa, genera un file di log strutturato che mostra la costruzione del JSON token per token.

### B.7 Argomenti annidati complessi
- Estendi lo schema per supportare parametri di tipo `object` o `array`.
- Modifica la FSM per gestire una pila di contesti: quando si apre una parentesi, si impila lo stato attuale e si inizia un nuovo contesto; quando si chiude, si ripristina lo stato precedente.
- Questo richiede un automa a pila (pushdown automaton) invece di una semplice FSM.

### B.8 Implementazione pubblica di encode/decode del tokenizer
- Realizza le funzioni `custom_encode` e `custom_decode` descritte in B.2 come parte della tua libreria interna, documentandole e rendendole riutilizzabili (non solo per il function calling).

---