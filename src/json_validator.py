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

    text: str
    status: State = State.START
    size: int = 0
    _stack: list[str] = PrivateAttr(default_factory=list)


    def model_post_init(self, __context) -> None:
        self.size = len(self.text)

    def _handle_string(self, i: int, next_state: State) -> tuple[int, State]:
        while i < len(self.text):
            if self.text[i] == "\"":
                return i + 1, next_state
            elif self.text[i] == "\\":
                if i + 1 >= len(self.text):
                    return (i, State.BAD)
                
                next_char = self.text[i + 1]
                
                if next_char in "\"\\/bfnrt":
                    i += 2
                elif next_char == "u":
                    hex_digits = self.text[i + 2:i + 6]
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
                if ord(self.text[i]) < 0x20:
                    return (i, State.BAD)
                i += 1
        if next_state == State.EXPECT_COLON:
            return i, State.IN_KEY_STRING
        else:
            return i, State.IN_STRING

    def _handle_number(self, i: int, next_state: State) -> tuple[int, State]:
        if i < self.size and self.text[i] == '-' and self.text[i] != '0':
            i += 1
        if self.text[i] == '0' and i + 1 < self.size and self.text[i + 1] in self._DIGITS:
            return (i, State.BAD)
        if i >= self.size or not self.text[i] in self._DIGITS:
            return (i, State.BAD)
        while i < self.size and self.text[i] in self._DIGITS:
            i += 1
        if i < self.size and self.text[i] == '.':
            i += 1
            if i >= self.size or not self.text[i] in self._DIGITS:
                return (i, State.IN_NUMBER)
            while i < self.size and self.text[i] in self._DIGITS:
                i += 1
        if i < self.size and self.text[i] in "Ee":
            i += 1
            if i < self.size and self.text[i] in "+-":
                i += 1
            if i >= self.size or not self.text[i] in self._DIGITS:
                return (i, State.IN_NUMBER)
            while i < self.size and self.text[i] in self._DIGITS:
                i += 1
        return (i, next_state)

    def _handle_special(self, i: int, next_state: State) -> tuple[int, State]:
        remaining = self.text[i:]
        for word, length in (("false", 5), ("null", 4), ("true", 4)):
            if remaining.startswith(word):
                return (i + length, next_state)

        for word in ("false", "null", "true"):
            if word.startswith(remaining):
                return (i + len(remaining), State.IN_SPECIAL)

        return (i, State.BAD)

    def _after_close(self) -> State:
        return State.END if not self._stack else State.EXPECT_COMMA_OR_END

    def validate(self) -> State:
        self.status = State.START
        self._stack.clear()
        i = 0
        while i < self.size:
 
            if self.text[i] in " \t\n\r" and self.status not in (State.IN_STRING, State.IN_KEY_STRING):
                i += 1
                continue

            if self.status == State.BAD:
                break

            elif self.status == State.START:
                if self.text[i] == "{":
                    self.status = State.EXPECT_KEY_OR_END_OBJ
                    self._stack.append("{")
                    i += 1
                elif self.text[i] == "[":
                    self.status = State.EXPECT_VALUE
                    self._stack.append("[")
                    i += 1
                else:
                    self.status = State.BAD

            elif self.status == State.IN_KEY_STRING:
                i, self.status = self._handle_string(i, State.EXPECT_COLON)

            elif self.status == State.EXPECT_COLON:
                if self.text[i] != ":":
                    self.status = State.BAD
                else:
                    self.status = State.EXPECT_VALUE
                    i += 1

            elif self.status == State.EXPECT_VALUE:
                allow_close = bool(self._stack) and self._stack[-1] == "["

                if self.text[i] == "\"":
                    self.status = State.IN_STRING
                    i += 1

                elif self.text[i] in self._DIGITS or self.text[i] == "-":
                    self.status = State.IN_NUMBER

                elif self.text[i] in "nft":
                    self.status = State.IN_SPECIAL

                elif self.text[i] == "[":
                    self.status = State.EXPECT_VALUE
                    self._stack.append("[")
                    i += 1

                elif self.text[i] == "{":
                    self.status = State.EXPECT_KEY_OR_END_OBJ
                    self._stack.append("{")
                    i += 1

                elif allow_close and self.text[i] == "]":
                    self._stack.pop()
                    self.status = self._after_close()
                    i += 1

                else:
                    self.status = State.BAD
            
            elif self.status == State.IN_STRING:
               i, self.status = self._handle_string(i, State.EXPECT_COMMA_OR_END)

            elif self.status == State.IN_NUMBER:
                i, self.status = self._handle_number(i, State.EXPECT_COMMA_OR_END)

            elif self.status == State.IN_SPECIAL:
                i, self.status = self._handle_special(i, State.EXPECT_COMMA_OR_END)
            
            elif self.status == State.EXPECT_KEY_OR_END_OBJ:
                if self.text[i] == "\"":
                    self.status = State.IN_KEY_STRING
                    i += 1

                elif self.text[i] in "}]" and self._stack and self._stack[-1] == "{":
                    if self._MAP_BRACKET[self.text[i]] == self._stack[-1][0]:
                        self._stack.pop()
                        self.status = self._after_close()
                        i += 1
                    else: self.status = State.BAD

                else: self.status = State.BAD

            elif self.status == State.EXPECT_COMMA_OR_END:
                if self.text[i] in "]}" and self._stack:
                    if self._MAP_BRACKET[self.text[i]] == self._stack[-1][0]:
                        self._stack.pop()
                        self.status = self._after_close()
                        i += 1
                    else:
                        self.status = State.BAD

                elif self.text[i] == ",":
                    if not self._stack:
                        self.status = State.BAD
                    elif self._stack[-1][0] == "{":
                        self._stack[-1] = "{,"
                        self.status = State.EXPECT_KEY_OR_END_OBJ
                        i += 1
                    else:
                        self._stack[-1] = "[,"
                        self.status = State.EXPECT_VALUE
                        i += 1
                else:
                    self.status = State.BAD
            
            else:
                self.status = State.BAD
        return self.status

    def next_char(self) -> set[str]:

        match self.status:
            case State.START:
                return {"{", "["}

            case State.EXPECT_KEY_OR_END_OBJ:
                if self._stack and self._stack[-1] == "{":
                    return {'"', "}"}
                return {'"'}

            case State.IN_KEY_STRING | State.IN_STRING:
                return self._STRING_CHARS | {'"', "\\"}

            case State.EXPECT_COLON:
                return {":"}

            case State.EXPECT_VALUE:
                allow_close = bool(self._stack) and self._stack[-1] == "["
                base = {'"', "-", "[", "{", "n", "t", "f"} | self._DIGITS
                if allow_close:
                    base.add("]")
                return base

            case State.IN_NUMBER:
                return self._DIGITS | {".", "e", "E", "+", "-", ",", "]", "}"}

            case State.IN_SPECIAL:
                return set("ruealsn")

            case State.EXPECT_COMMA_OR_END:
                if self._stack and self._stack[-1][0] == "{":
                    return {",", "}"}
                elif self._stack and self._stack[-1][0] == "[":
                    return {",", "]"}
                return {",", "]", "}"}

            case State.END | State.BAD:
                return set()

            case _:
                return set()

    def is_complete(self) -> bool:
        return self.validate() == State.END