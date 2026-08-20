import string
from pydantic import BaseModel, PrivateAttr
from enum import Enum, auto
from typing import ClassVar


class State(Enum):
    START = auto()
    EXPECT_KEY_OR_END_OBJ = auto()
    IN_KEY_STRING = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    IN_SPECIAL = auto()
    EXPECT_COMMA_OR_END = auto()
    END = auto()
    BAD = auto()



class JsonValidator(BaseModel):
    _MAP_BRACKET: ClassVar[dict[str, str]] = {"}": "{", "]": "["}
    _DIGITS : ClassVar[set[str]] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    _STRING_CHARS: ClassVar[set[str]] = {
        chr(c) for c in range(0x20, 0x110000)
    } - {'"', "\\"}
    # Caratteri che possono comparire ALL'INTERNO di un numero o di un literal
    # speciale (true/false/null). Usati per ritrovare, con un backward-scan,
    # l'indice di INIZIO del token corrente quando si riprende il parsing in
    # modo incrementale da un punto intermedio.
    _NUMBER_CHARS: ClassVar[set[str]] = _DIGITS | {".", "e", "E", "+", "-"}
    _SPECIAL_CHARS: ClassVar[set[str]] = set("truefalsn")

    text: str
    _status: State = PrivateAttr(default=State.START)
    _stack: list[str] = PrivateAttr(default_factory=list)
    _key_stack: list[str | None] = PrivateAttr(default_factory=list)

    @property
    def status(self) -> State:
        return self._status
    
    @property
    def stack(self) -> list[str]:
        return self._stack

    def _handle_string(self, i: int, text: str, next_state: State) -> tuple[int, State]:
        while i < len(text):
            if text[i] == "\"":
                return i + 1, next_state
            elif text[i] == "\\":
                if i + 1 >= len(text):
                    return (i, State.BAD)
                
                next_char = text[i + 1]
                
                if next_char in "\"\\/bfnrt":
                    i += 2
                elif next_char == "u":
                    hex_digits = text[i + 2:i + 6]
                    if len(hex_digits) != 4:
                        return (i, State.BAD)
                    try:
                        int(hex_digits, 16)
                    except ValueError:
                        return (i, State.BAD)
                    i += 6
                else:
                    return (i, State.BAD)
            else:
                if ord(text[i]) < 0x20:
                    return (i, State.BAD)
                i += 1
        if next_state == State.EXPECT_COLON:
            return i, State.IN_KEY_STRING
        else:
            return i, State.IN_STRING

    def _handle_number(self, i: int, text: str, next_state: State) -> tuple[int, State]:
        size = len(text)
        if i < size and text[i] == '-':
            i += 1
        if i >= size or text[i] not in self._DIGITS:
            return (i, State.BAD)
        if text[i] == '0' and i + 1 < size and text[i + 1] in self._DIGITS:
            return (i, State.BAD)
        while i < size and text[i] in self._DIGITS:
            i += 1
        if i < size and text[i] == '.':
            i += 1
            if i >= size or not text[i] in self._DIGITS:
                return (i, State.IN_NUMBER)
            while i < size and text[i] in self._DIGITS:
                i += 1
        if i < size and text[i] in "Ee":
            i += 1
            if i < size and text[i] in "+-":
                i += 1
            if i >= size or not text[i] in self._DIGITS:
                return (i, State.IN_NUMBER)
            while i < size and text[i] in self._DIGITS:
                i += 1
        if i == size:
            return (i, State.IN_NUMBER)
        return (i, next_state)

    def _handle_special(self, i: int, text: str, next_state: State) -> tuple[int, State]:
        remaining = text[i:]
        for word, length in (("false", 5), ("null", 4), ("true", 4)):
            if remaining.startswith(word):
                return (i + length, next_state)

        for word in ("false", "null", "true"):
            if word.startswith(remaining):
                return (i + len(remaining), State.IN_SPECIAL)

        return (i, State.BAD)

    def _after_close(self, stack: list[str]) -> State:
        return State.END if not stack else State.EXPECT_COMMA_OR_END

    def _find_token_start(self, text: str, status: State, hint: int) -> int:
        """Trova l'indice di INIZIO del token corrente (numero, literal
        speciale true/false/null, o chiave stringa) scandendo `text` a
        ritroso a partire da `hint`.

        Necessario perche' _handle_number/_handle_special (e l'estrazione
        della chiave in IN_KEY_STRING) assumono che l'indice passato punti
        all'inizio del token: riprendendo il parsing da un punto intermedio
        (parsing incrementale) senza questa correzione si perde il contesto
        necessario per applicare correttamente le regole di validita' (es.
        "0" seguito da un'altra cifra e' invalido, "nn" non e' un prefisso
        valido di alcun literal).
        """
        if status == State.IN_NUMBER:
            j = hint
            while j > 0 and text[j - 1] in self._NUMBER_CHARS:
                j -= 1
            return j
        if status == State.IN_SPECIAL:
            j = hint
            while j > 0 and text[j - 1] in self._SPECIAL_CHARS:
                j -= 1
            return j
        if status == State.IN_KEY_STRING:
            j = text.rfind('"', 0, hint)
            return j + 1 if j != -1 else hint
        return hint

    def _run(
        self,
        text: str,
        status: State=State.START,
        stack: list[str] | None = None,
        key_stack: list[str | None] | None = None,
        start: int=0
        ) -> tuple[State, list[str], list[str | None], int]:
        i = start
        size = len(text)
        if stack is None:
            stack = []
        if key_stack is None:
            key_stack = []
        while i < size:
 
            if text[i] in " \t\n\r" and status not in (State.IN_STRING, State.IN_KEY_STRING):
                i += 1
                continue

            if status == State.BAD:
                break

            elif status == State.START:
                if text[i] == "{":
                    status = State.EXPECT_KEY_OR_END_OBJ
                    stack.append("{")
                    key_stack.append(None)
                    i += 1
                elif text[i] == "[":
                    status = State.EXPECT_VALUE
                    stack.append("[")
                    key_stack.append(None)
                    i += 1
                else:
                    status = State.BAD

            elif status == State.IN_KEY_STRING:
                start_key = i
                i, status = self._handle_string(i, text, State.EXPECT_COLON)
                if status == State.EXPECT_COLON:
                    key_stack[-1] = text[start_key:i-1]
            elif status == State.EXPECT_COLON:
                if text[i] != ":":
                    status = State.BAD
                else:
                    status = State.EXPECT_VALUE
                    i += 1

            elif status == State.EXPECT_VALUE:
                allow_close = bool(stack) and stack[-1] == "["

                if text[i] == "\"":
                    status = State.IN_STRING
                    i += 1

                elif text[i] in self._DIGITS or text[i] == "-":
                    status = State.IN_NUMBER

                elif text[i] in "nft":
                    status = State.IN_SPECIAL

                elif text[i] == "[":
                    status = State.EXPECT_VALUE
                    stack.append("[")
                    key_stack.append(None)
                    i += 1

                elif text[i] == "{":
                    status = State.EXPECT_KEY_OR_END_OBJ
                    stack.append("{")
                    key_stack.append(None)
                    i += 1

                elif allow_close and text[i] == "]":
                    stack.pop()
                    key_stack.pop()
                    status = self._after_close(stack)
                    i += 1

                else:
                    status = State.BAD
            
            elif status == State.IN_STRING:
               i, status = self._handle_string(i, text, State.EXPECT_COMMA_OR_END)

            elif status == State.IN_NUMBER:
                i, status = self._handle_number(i, text, State.EXPECT_COMMA_OR_END)

            elif status == State.IN_SPECIAL:
                i, status = self._handle_special(i, text, State.EXPECT_COMMA_OR_END)
            
            elif status == State.EXPECT_KEY_OR_END_OBJ:
                if text[i] == "\"":
                    status = State.IN_KEY_STRING
                    i += 1

                elif text[i] in "}]" and stack and stack[-1] == "{":
                    if self._MAP_BRACKET[text[i]] == stack[-1][0]:
                        stack.pop()
                        key_stack.pop()
                        status = self._after_close(stack)
                        i += 1
                    else: status = State.BAD

                else: status = State.BAD

            elif status == State.EXPECT_COMMA_OR_END:
                if text[i] in "]}" and stack:
                    if self._MAP_BRACKET[text[i]] == stack[-1][0]:
                        stack.pop()
                        key_stack.pop()
                        status = self._after_close(stack)
                        i += 1
                    else:
                        status = State.BAD

                elif text[i] == ",":
                    if not stack:
                        status = State.BAD
                    elif stack[-1][0] == "{":
                        stack[-1] = "{,"
                        key_stack[-1] = None 
                        status = State.EXPECT_KEY_OR_END_OBJ
                        i += 1
                    else:
                        stack[-1] = "[,"
                        status = State.EXPECT_VALUE
                        i += 1
                else:
                    status = State.BAD
            
            else:
                status = State.BAD
        return status, stack, key_stack, i

    def advance(
        self,
        text: str,
        status: State,
        stack: list[str],
        key_stack: list,
        extra_text: str,
        ) -> tuple[State, list[str], list, str]:
        candidate = text + extra_text
        # Se stiamo riprendendo da dentro un token multi-carattere (numero,
        # literal speciale, o chiave stringa), ritroviamo l'inizio vero del
        # token invece di ripartire dal punto intermedio in cui eravamo
        # rimasti: altrimenti _handle_number/_handle_special perdono il
        # contesto necessario per validare correttamente (vedi
        # _find_token_start).
        resume_start = self._find_token_start(text, status, len(text))
        new_status, new_stack, new_key_stack, _ = self._run(
            candidate,
            status=status,
            stack=list(stack),
            key_stack=list(key_stack),
            start=resume_start,
        )
        return new_status, new_stack, new_key_stack, candidate

    def validate(self) -> State:
        # NON usare più per l'aggiornamento incrementale in consume().
        # Resta utile solo se vuoi rivalidare da zero per qualche motivo (debug, sanity check).
        status, stack, key_stack, _ = self._run(self.text)
        self._status = status
        self._stack = stack
        self._key_stack = key_stack
        return status

    def consume_incremental(self, extra_text: str) -> None:
        """Aggiorna lo stato in modo incrementale invece di rifare il parsing da zero.

        Delega ad advance(), che gia' gestisce correttamente sia il caso
        "riprendo da uno stato strutturale" (safe, start=len(text)) sia il
        caso "riprendo da dentro un token" (backward-scan del vero inizio
        del token, vedi _find_token_start).
        """
        new_status, new_stack, new_key_stack, new_text = self.advance(
            text=self.text,
            status=self._status,
            stack=self._stack,
            key_stack=self._key_stack,
            extra_text=extra_text,
        )
        self.text = new_text
        self._status = new_status
        self._stack = new_stack
        self._key_stack = new_key_stack

    def is_complete(self) -> bool:
        return self.validate() == State.END

    def next_char(
        self,
        status: State | None = None,
        stack: list[str] | None = None
        ) -> set[str]:

        status = self._status if status is None else status
        stack = self._stack if stack is None else stack
        
        whitespaces = {" ", "\n", "\t", "\r"}
        
        match status:
            case State.START:
                return {"{", "["} | whitespaces

            case State.EXPECT_KEY_OR_END_OBJ:
                base = {'"'} | whitespaces
                if stack and stack[-1] == "{":
                    base.add("}")
                return base

            case State.IN_KEY_STRING | State.IN_STRING:
                return self._STRING_CHARS | {'"', "\\"}

            case State.EXPECT_COLON:
                return {":"} | whitespaces

            case State.EXPECT_VALUE:
                allow_close = bool(stack) and stack[-1] == "["
                base = {'"', "-", "[", "{", "n", "t", "f"} | self._DIGITS | whitespaces
                if allow_close:
                    base.add("]")
                return base

            case State.IN_NUMBER:
                return self._DIGITS | {".", "e", "E", "+", "-", ",", "]", "}"} | whitespaces

            case State.IN_SPECIAL:
                return set("ruealsn")

            case State.EXPECT_COMMA_OR_END:
                base = {",", "]", "}"} | whitespaces
                if stack and stack[-1][0] == "{":
                    base = {",", "}"} | whitespaces
                elif stack and stack[-1][0] == "[":
                    base = {",", "]"} | whitespaces
                return base

            case State.END | State.BAD:
                return set()

            case _:
                return set()

    def is_valid_continuation(self, token_str: str) -> bool:
        new_status, _, _, _ = self.advance(
            text=self.text,
            status=self._status,
            stack=list(self._stack),
            key_stack=list(self._key_stack),
            extra_text=token_str,
        )
        return new_status != State.BAD

    def current_key_path(self) -> list[str | None]:
        return list(self._key_stack)