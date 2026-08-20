from pydantic import BaseModel, PrivateAttr
from .json_validator import JsonValidator, State
from .schema_aware_validator import SchemaAwareValidator


NUMBER_START_CHARS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"}
NULL_START_CHARS = {"n"}
BOOLEAN_START_CHARS = {"t", "f"}

class SchemaConstrainedDecoder(BaseModel):
    validator: JsonValidator
    func_schema: SchemaAwareValidator
    _resolved_function_name : str | None = PrivateAttr(default=None)

    @property
    def resolved_function_name(self) -> str | None:
        return self._resolved_function_name

    def allowed_next_chars(
        self,
        text: str | None = None,
        status: State | None = None,
        stack: list | None = None,
        path: list | None = None,
        ) -> set[str]:
        
        text = text if text is not None else self.validator.text
        status = status if status is not None else self.validator.status
        stack = stack if stack is not None else self.validator.stack
        path = path if path is not None else self.validator.current_key_path()

        syntatic_allowed = set(self.validator.next_char(status=status, stack=stack))
        WHITESPACES = {" ", "\n", "\t", "\r"}

        # --- 1. GESTIONE RADICE: Forza l'ordine stringente "name" -> "parameters" ---
        if len(path) == 1:
            if self.resolved_function_name is None:
                # Non abbiamo ancora il nome: VIETA la chiusura dell'oggetto '{'
                if status == State.EXPECT_KEY_OR_END_OBJ:
                    syntatic_allowed.discard("}")
                
                # Forza esclusivamente la chiave "name"
                if path[-1] is None and status == State.IN_KEY_STRING:
                    start = text.rfind('"')
                    partial = text[start+1:]
                    valid_chars = set()
                    for k in {"name"}:
                        if k.startswith(partial):
                            if len(k) > len(partial):
                                valid_chars.add(k[len(partial)])
                            elif len(k) == len(partial):
                                valid_chars.add('"')
                    return syntatic_allowed & valid_chars
            else:
                # Abbiamo il nome, ora forziamo la chiave "parameters"
                if path[-1] == "name" and status == State.EXPECT_COMMA_OR_END:
                    syntatic_allowed.discard("}") # Vieta di chiudere senza i parametri
                
                if path[-1] is None and status == State.IN_KEY_STRING:
                    start = text.rfind('"')
                    partial = text[start+1:]
                    valid_chars = set()
                    for k in {"parameters"}:
                        if k.startswith(partial):
                            if len(k) > len(partial):
                                valid_chars.add(k[len(partial)])
                            elif len(k) == len(partial):
                                valid_chars.add('"')
                    return syntatic_allowed & valid_chars

        # --- 2. VALORE DI "name" ---
        if path and path[-1] == "name" and status == State.IN_STRING:
            start = text.rfind('"')
            partial = text[start+1:]
            schema_allowed = self.func_schema.next_char_for_name(partial)
            if self.func_schema.is_complete(partial):
                schema_allowed.add('"')
            return syntatic_allowed & schema_allowed

        # --- 3. CHIAVI DEI PARAMETRI ---
        if len(path) == 2 and path[0] == "parameters" and status == State.IN_KEY_STRING:
            if self.resolved_function_name is None:
                return syntatic_allowed
            start = text.rfind('"')
            partial = text[start+1:]
            schema_allowed = self.func_schema.next_char_for_param(
                self.resolved_function_name, partial
            )
            if self.func_schema.is_complete_for_param(self.resolved_function_name, partial):
                schema_allowed.add('"')
            return syntatic_allowed & schema_allowed
            
        # --- 4. VALORI DEI PARAMETRI ---
        if len(path) == 2 and path[0] == "parameters" and status == State.EXPECT_VALUE:
            param = path[1]
            if self.resolved_function_name is None:
                return syntatic_allowed
            param_type = self.func_schema.get_param_type(
                self.resolved_function_name, param
            )
            
            if param_type == "number":
                return syntatic_allowed & (NUMBER_START_CHARS | WHITESPACES)
            elif param_type == "boolean":
                return syntatic_allowed & (BOOLEAN_START_CHARS | WHITESPACES)
            elif param_type == "null":
                return syntatic_allowed & (NULL_START_CHARS | WHITESPACES)
            elif param_type == "string":
                return syntatic_allowed & ({'"'} | WHITESPACES)
            elif param_type == "array":
                return syntatic_allowed & ({'['} | WHITESPACES)
            elif param_type == "object":
                return syntatic_allowed & ({'{'} | WHITESPACES)
                
        return syntatic_allowed

    def consume(self, token_str: str) -> None:
        self.validator.consume_incremental(token_str)
        
        path = self.validator.current_key_path()
        status = self.validator.status
        
        # Se abbiamo superato la chiave "name" ed elaborato il suo valore
        if len(path) == 1 and path[0] == "name" and status in (State.EXPECT_COMMA_OR_END, State.EXPECT_KEY_OR_END_OBJ):
            text = self.validator.text
            end = text.rfind('"')
            start = text.rfind('"', 0, end)
            if start != -1:
                self._resolved_function_name = text[start + 1:end]

    def advance_hypothetical(
        self,
        text: str,
        status: State,
        stack: list[str],
        path: list, 
        extra_char: str,
        ) -> tuple[str, State, list, list]:

        new_status, new_stack, new_key_stack, new_text = self.validator.advance(
            text=text,
            status=status,
            stack=stack,
            key_stack=path,
            extra_text=extra_char,
        )
        return new_text, new_status, new_stack, new_key_stack