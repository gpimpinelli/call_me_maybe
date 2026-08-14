import json
from pydantic import BaseModel, PrivateAttr, Field
from enum import Enum, auto

# Manca
# IN_ARRAY
# EXPECT_VALUE_OR_END_ARRAY
# EXPECT_COMMA_OR_END_ARRAY
# EXPECT_KEY_OR_END_OBJ

class State(Enum):
    START = auto()
    EXPECT_KEY_OR_END_OBJ = auto()
    IN_KEY_STRING = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    IN_SPECIAL = auto()
    IN_ARRAY = auto()
    EXPECT_COMMA_OR_END_OBJ = auto()
    EXPECT_VALUE_OR_END_ARRAY = auto()
    EXPECT_COMMA_OR_END_ARRAY = auto()
    END = auto()
    BAD = auto()



class JsonValidator(BaseModel):
    text: str
    status: State = State.START
    size: int = 0

    def model_post_init(self, __context) -> None:
        self.size = len(self.text)

    def handle_string(self, i: int, next_state: State) -> tuple[int, State]:
        i += 1
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
                i += 1
        if next_state == State.EXPECT_COLON:
            return i, State.IN_KEY_STRING
        else:
            return i, State.IN_STRING

    def handle_number(self, i: int) -> tuple[int, State]:
        if i < self.size and self.text[i] == '-':
            i += 1
        if i >= self.size or not self.text[i].isdigit():
            return (i, State.BAD)
        while i < self.size and self.text[i].isdigit():
            i += 1
        if i < self.size and self.text[i] == '.':
            i += 1
            if i >= self.size or not self.text[i].isdigit():
                return (i, State.IN_NUMBER)
            while i < self.size and self.text[i].isdigit():
                i += 1
        if i < self.size and self.text[i] in "Ee":
            i += 1
            if i + 1 < self.size and self.text[i + 1] in "+-":
                i += 1
            if i >= self.size or not self.text[i].isdigit():
                return (i, State.IN_NUMBER)
            while i < self.size and self.text[i].isdigit():
                i += 1
        return (i, State.EXPECT_COMMA_OR_END_OBJ)

    def handle_special(self, i: int) -> tuple[int, State]:
        remaining = self.text[i:]

        for word, length in (("false", 5), ("null", 4), ("true", 4)):
            if remaining.startswith(word):
                return (i + length, State.EXPECT_COMMA_OR_END_OBJ)

        for word in ("false", "null", "true"):
            if word.startswith(remaining):
                return (i + len(remaining), State.IN_SPECIAL)

        return (i, State.BAD)

    def validate(self) -> State:
        i = 0

        while i < self.size:
            if self.status == State.BAD:
                break
 
            if self.text[i].isspace() and self.status != State.IN_STRING:
                i += 1

            elif self.status == State.START:
                if self.text[i] == "{":
                    self.status = State.EXPECT_KEY_OR_END_OBJ
                    i += 1
                else:
                    self.status = State.BAD

            elif self.status == State.IN_KEY_STRING:
                res = self.handle_string(i, State.EXPECT_COLON)
                i, self.status = res

            elif self.status == State.EXPECT_COLON:
                if self.text[i] != ":":
                    self.status = State.BAD
                else:
                    self.status = State.EXPECT_VALUE
                    i += 1

            elif self.status == State.EXPECT_VALUE:
                if self.text[i] == "\"":
                    res = self.handle_string(i, State.EXPECT_COMMA_OR_END_OBJ)
                    i, self.status = res

                elif self.text[i].isdigit():
                    self.status = State.IN_NUMBER
                
                elif self.text[i] in "nft":
                    self.status = State.IN_SPECIAL
                
                elif self.text[i] == "[":
                    self.status = State.IN_ARRAY
                    i += 1
                
                elif self.text[i] == "{":
                    self.status = State.EXPECT_KEY_OR_END_OBJ
                    i += 1
                else: self.status = State.BAD

            elif self.status == State.IN_NUMBER:
                res = self.handle_number(i)
                i, self.status = res

            elif self.status == State.IN_SPECIAL:
                res = self.handle_special(i)
                i, self.status = res

            elif self.status == State.EXPECT_COMMA_OR_END_OBJ:
                if self.text[i] == "}":
                    self.status = State.END
                    i += 1
                elif self.text[i] == ",":
                    self.status = State.EXPECT_KEY_OR_END_OBJ
                    i += 1
                else: self.status = State.BAD
            else:
                self.status = State.BAD
        return self.status