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
    _NUMBER_CHARS: ClassVar[set[str]] = _DIGITS | {".", "e", "E", "+", "-"}
    _SPECIAL_CHARS: ClassVar[set[str]] = set("truefalsn")
    # --- ECCO LA RIGA MANCANTE CHE RISOLVE IL CRASH ---


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

    def _step(
        self,
        status: State,
        stack: list[str],
        key_stack: list[str | None],
        current_text: str,
        char: str
    ) -> tuple[State, list[str], list[str | None]]:
        # Fast path per gli spazi bianchi (fuori dalle stringhe)
        if char in " \t\n\r" and status not in (State.IN_STRING, State.IN_KEY_STRING):
            if status in (State.IN_NUMBER, State.IN_SPECIAL):
                return State.EXPECT_COMMA_OR_END, stack, key_stack
            return status, stack, key_stack

        if status == State.START:
            if char == "{":
                return State.EXPECT_KEY_OR_END_OBJ, stack + ["{"], key_stack + [None]
            elif char == "[":
                return State.EXPECT_VALUE, stack + ["["], key_stack + [None]
            return State.BAD, stack, key_stack

        elif status == State.EXPECT_KEY_OR_END_OBJ:
            if char == '"':
                return State.IN_KEY_STRING, stack, key_stack
            elif char in "}]" and stack and stack[-1] == "{":
                if self._MAP_BRACKET[char] == stack[-1][0]:
                    return (State.END if len(stack) == 1 else State.EXPECT_COMMA_OR_END), stack[:-1], key_stack[:-1]
            return State.BAD, stack, key_stack

        elif status == State.IN_KEY_STRING:
            if char == '"':
                # Estrazione istantanea della chiave senza ricalcolare tutto
                start = current_text.rfind('"')
                key_name = current_text[start+1:]
                return State.EXPECT_COLON, stack, key_stack[:-1] + [key_name]
            elif char == '\\':
                return State.IN_KEY_STRING, stack, key_stack
            elif ord(char) < 0x20:
                return State.BAD, stack, key_stack
            return State.IN_KEY_STRING, stack, key_stack

        elif status == State.EXPECT_COLON:
            if char == ":":
                return State.EXPECT_VALUE, stack, key_stack
            return State.BAD, stack, key_stack

        elif status == State.EXPECT_VALUE:
            allow_close = bool(stack) and stack[-1] == "["
            if char == '"':
                return State.IN_STRING, stack, key_stack
            elif char in self._DIGITS or char == "-":
                return State.IN_NUMBER, stack, key_stack
            elif char in "nft":
                return State.IN_SPECIAL, stack, key_stack
            elif char == "{":
                return State.EXPECT_KEY_OR_END_OBJ, stack + ["{"], key_stack + [None]
            elif char == "[":
                return State.EXPECT_VALUE, stack + ["["], key_stack + [None]
            elif allow_close and char == "]":
                return (State.END if len(stack) == 1 else State.EXPECT_COMMA_OR_END), stack[:-1], key_stack[:-1]
            return State.BAD, stack, key_stack

        elif status == State.IN_STRING:
            if char == '"':
                return State.EXPECT_COMMA_OR_END, stack, key_stack
            elif char == '\\':
                return State.IN_STRING, stack, key_stack
            elif ord(char) < 0x20:
                return State.BAD, stack, key_stack
            return State.IN_STRING, stack, key_stack

        elif status in (State.IN_NUMBER, State.IN_SPECIAL):
            valid_chars = self._NUMBER_CHARS if status == State.IN_NUMBER else self._SPECIAL_CHARS
            if char in valid_chars:
                return status, stack, key_stack
            elif char in "]}":
                if stack and self._MAP_BRACKET.get(char) == stack[-1][0]:
                    return (State.END if len(stack) == 1 else State.EXPECT_COMMA_OR_END), stack[:-1], key_stack[:-1]
                return State.BAD, stack, key_stack
            elif char == ",":
                if not stack:
                    return State.BAD, stack, key_stack
                if stack[-1][0] == "{":
                    return State.EXPECT_KEY_OR_END_OBJ, stack[:-1] + ["{,"], key_stack[:-1] + [None]
                else:
                    return State.EXPECT_VALUE, stack[:-1] + ["[,"], key_stack
            return State.BAD, stack, key_stack

        elif status == State.EXPECT_COMMA_OR_END:
            if char in "]}":
                if stack and self._MAP_BRACKET.get(char) == stack[-1][0]:
                    return (State.END if len(stack) == 1 else State.EXPECT_COMMA_OR_END), stack[:-1], key_stack[:-1]
                return State.BAD, stack, key_stack
            elif char == ",":
                if not stack:
                    return State.BAD, stack, key_stack
                if stack[-1][0] == "{":
                    return State.EXPECT_KEY_OR_END_OBJ, stack[:-1] + ["{,"], key_stack[:-1] + [None]
                else:
                    return State.EXPECT_VALUE, stack[:-1] + ["[,"], key_stack
            return State.BAD, stack, key_stack

        return State.BAD, stack, key_stack

    def advance(
        self,
        text: str,
        status: State,
        stack: list[str],
        key_stack: list[str | None],
        extra_text: str,
    ) -> tuple[State, list[str], list, str]:
        
        new_status = status
        new_stack = stack
        new_key_stack = key_stack
        current_text = text

        for char in extra_text:
            if new_status == State.BAD:
                break
            
            new_status, new_stack, new_key_stack = self._step(
                new_status, new_stack, new_key_stack, current_text, char
            )
            current_text += char

        return new_status, new_stack, new_key_stack, current_text

    def validate(self) -> State:
        status = State.START
        stack = []
        key_stack = []
        current_text = ""
        for char in self.text:
            if status == State.BAD:
                break
            status, stack, key_stack = self._step(status, stack, key_stack, current_text, char)
            current_text += char
            
        self._status = status
        self._stack = stack
        self._key_stack = key_stack
        return status

    def consume_incremental(self, extra_text: str) -> None:
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