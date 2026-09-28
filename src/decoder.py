import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


_FORBIDDEN_STRING_CHARS = frozenset('"\\')

_NUMBER_CHARS = "0123456789.+-eE"

_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


def build_context(functions: Sequence[Any]) -> str:
    """
    Format available function definitions into a system prompt.
    """
    header = (
        "You are a function-calling assistant.\n"
        "Choose exactly one function.\n"
        "Output only the function name.\n\n"
        "Available functions:"
    )

    blocks = [header]
    for fn in functions:
        params_str = "\n".join(
            f"  - {name}: {param.type}"
            for name, param in fn.parameters.items()
        )
        blocks.append(
            f"Function: {fn.name}\n"
            f"Description: {fn.description}\n"
            f"Parameters:\n{params_str}\n"
            f"Returns: {fn.returns.type}"
        )

    return "\n\n".join(blocks) + "\n"


def _number_step(state: str, ch: str) -> Optional[str]:
    """
    Advance the JSON number grammar state machine by one character.
    """
    is_dig = ch.isdigit()

    match state, ch:
        case "start", "-":
            return "after_minus"
        case "start" | "after_minus", "0":
            return "zero"
        case "start" | "after_minus", _ if is_dig:
            return "int"
        case "zero" | "int", ".":
            return "dot"
        case "zero" | "int" | "frac", "e" | "E":
            return "exp"
        case "int" | "frac", _ if is_dig:
            return state
        case "dot", _ if is_dig:
            return "frac"
        case "exp", "+" | "-":
            return "exp_sign"
        case "exp" | "exp_sign" | "exp_digit", _ if is_dig:
            return "exp_digit"
        case _:
            return None


def _number_done(state: str) -> bool:
    """Tell whether a number could legally end in this state."""
    return state in ("zero", "int", "frac", "exp_digit")


def _number_candidates(state: str, integer_only: bool) -> str:
    """
    Retrieve all valid characters that can follow the current state.
    """
    match state:
        case "start":
            return "-0123456789"
        case "after_minus":
            return "0123456789"
        case "zero":
            return "" if integer_only else ".eE"
        case "int":
            return "0123456789" if integer_only else "0123456789.eE"
        case "dot" | "exp_sign" | "exp_digit":
            return "0123456789"
        case "frac":
            return "0123456789eE"
        case "exp":
            return "+-0123456789"
        case _:
            return ""


class Decoder:
    """
    Drives constrained generation by filtering model logits token by token.
    """
    def __init__(self, model: Any, functions: Sequence[Any]) -> None:
        self.model = model
        # for functions:
        self.functions = functions
        self.function_tokens = self.build_function_tokens(functions)
        # for parameters:
        self.function_parameters = self.build_function_parameters(functions)
        self.boolean_tokens: Dict[str, List[int]] = {
            "true": self.model.encode("true").squeeze(0).tolist(),
            "false": self.model.encode("false").squeeze(0).tolist(),
        }
        self._digit_token_ids = self._build_digit_token_ids()
        self._string_safe_ids_cache: Optional[np.ndarray] = None
        self._quote_id = self._maybe_token_id('"')
        self._backslash_id = self._maybe_token_id("\\")
        self._escape_token_ids = self._build_escape_token_ids()

    # --- for functions -------------------------------------------------

    def generate_function_name(self, prompt: str) -> str:

        name, _ = self._select_one_of(prompt, self.function_tokens)

        if name is None:
            # Every option starts eligible at step 0 (an empty prefix
            # matches every token list), so this only fires if the
            # model wandered outside max_new_tokens without finishing
            # any name. Fall back to the first catalog entry rather
            # than crash the whole batch over one bad prompt.
            name = next(iter(self.function_tokens))

        return name

    def _select_one_of(
        self,
        prompt_or_ids: Union[str, List[int]],
        token_map: Dict[str, List[int]],
        is_ids: bool = False,
        max_new_tokens: int = 20,
    ) -> Tuple[Optional[str], List[int]]:
        input_ids: List[int]
        if is_ids:
            assert isinstance(prompt_or_ids, list)
            input_ids = list(prompt_or_ids)
        else:
            assert isinstance(prompt_or_ids, str)
            input_ids = self.model.encode(prompt_or_ids).squeeze(0).tolist()

        generated: List[int] = []

        for _ in range(max_new_tokens):

            allowed = set()
            exact: Optional[str] = None

            for value, token_ids in token_map.items():

                if token_ids[: len(generated)] != generated:
                    continue

                if len(generated) < len(token_ids):
                    allowed.add(token_ids[len(generated)])
                elif exact is None:
                    exact = value

            if exact is not None:
                return exact, input_ids

            if not allowed:
                break

            logits = self.model.get_logits_from_input_ids(input_ids)

            best_token: Optional[int] = None
            best_logit: Optional[float] = None

            for token in allowed:
                if best_token is None or logits[token] > best_logit:
                    best_token = token
                    best_logit = logits[token]

            assert best_token is not None  # allowed is non-empty here

            generated.append(best_token)
            input_ids.append(best_token)

        return None, input_ids

    def build_function_tokens(
        self,
        functions: Sequence[Any],
    ) -> Dict[str, List[int]]:
        tokenized = {}

        for fn in functions:
            ids = self.model.encode(fn.name).squeeze(0).tolist()
            tokenized[fn.name] = ids

        return tokenized

    # --- for parameters --------------------------------------------

    def build_function_parameters(
        self,
        functions: Sequence[Any],
    ) -> Dict[str, Any]:
        parameters = {}

        for fn in functions:
            parameters[fn.name] = fn.parameters

        return parameters

    def token_id(self, text: str) -> int:
        """
        Get token id for a single token string.
        """
        ids = self.model.encode(text).squeeze(0).tolist()

        if len(ids) != 1:
            raise ValueError(f"'{text}' is not a single token: {ids}")

        return int(ids[0])

    def _maybe_token_id(self, text: str) -> Optional[int]:
        """Like token_id, but None instead of raising.

        Not every tokenizer necessarily spells every one of "+eE" as
        its own single token; the number grammar simply does not
        offer a character whose token does not exist, rather than
        fail the whole run over an exponent sign nobody asked for.
        """
        try:
            return self.token_id(text)
        except ValueError:
            return None

    def _build_digit_token_ids(self) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for ch in _NUMBER_CHARS:
            token_id = self._maybe_token_id(ch)
            if token_id is not None:
                ids[ch] = token_id
        return ids

    def _build_escape_token_ids(self) -> Dict[int, str]:
        """Map the token id of each escapable character to itself.

        Built once: the tokens right after a backslash in a JSON
        string are a closed set of eight characters, exactly like the
        digits of a number, so no vocabulary scan is needed here
        either.
        """
        ids: Dict[int, str] = {}
        for ch in _ESCAPE_MAP:
            token_id = self._maybe_token_id(ch)
            if token_id is not None:
                ids[token_id] = ch
        return ids

    def _encode_literal(self, text: str) -> List[int]:
        """Encode fixed JSON text the decoder writes itself.

        Nothing here is a model choice: punctuation, key names and
        quotes are always exactly what the catalog and the JSON
        grammar require, so they are written directly rather than
        offered to the model one token at a time.
        """
        if not text:
            return []
        ids: List[int] = self.model.encode(text).squeeze(0).tolist()
        return ids

    def generate_parameters(
        self,
        prompt: str,
        function_name: str,
    ) -> Dict[str, Any]:
        """
        Generate every argument of one function call under constrained
        decoding: punctuation and key names are forced literal text,
        and each value is generated by the model under the grammar of
        its declared type (string / number / integer / boolean).

        prompt is the same context used to choose the function name
        (system instructions, catalog, user request), so the model
        sees the request while it writes the values, not just the
        bare parameter list.
        """
        parameters = self.function_parameters[function_name]

        input_ids = self.model.encode(prompt).squeeze(0).tolist()
        input_ids += self._encode_literal('{"parameters": {')

        result: Dict[str, Any] = {}
        items = list(parameters.items())

        for index, (parameter_name, parameter) in enumerate(items):

            is_last = index == len(items) - 1
            closer = "}}" if is_last else ", "
            value: Any

            input_ids += self._encode_literal(f'"{parameter_name}": ')

            if parameter.type == "string":
                input_ids += self._encode_literal('"')
                value, input_ids = self._generate_string_value(
                    input_ids,
                    closer_text='"' + closer,
                )

            elif parameter.type in ("number", "integer"):
                value, input_ids = self._generate_number_value(
                    input_ids,
                    closer_text=closer,
                    integer_only=parameter.type == "integer",
                )

            elif parameter.type == "boolean":
                value, input_ids = self._generate_boolean_value(
                    input_ids,
                    closer_text=closer,
                )

            else:
                value = None
                input_ids += self._encode_literal(closer)

            result[parameter_name] = value

        return result

    def _generate_boolean_value(
        self,
        input_ids: List[int],
        closer_text: str,
    ) -> Tuple[Optional[bool], List[int]]:

        text, input_ids = self._select_one_of(
            input_ids,
            self.boolean_tokens,
            is_ids=True,
        )

        value: Optional[bool] = None
        if text is not None:
            value = {"true": True, "false": False}.get(text)

        input_ids = input_ids + self._encode_literal(closer_text)

        return value, input_ids

    def _generate_number_value(
        self,
        input_ids: List[int],
        closer_text: str,
        integer_only: bool,
        max_digits: int = 24,
    ) -> Tuple[Optional[Union[int, float]], List[int]]:

        closer_first_id = self.model.encode(
            closer_text
        ).squeeze(0).tolist()[0]

        state = "start"
        text = ""

        for _ in range(max_digits):

            candidates = {
                self._digit_token_ids[ch]: ch
                for ch in _number_candidates(state, integer_only)
                if ch in self._digit_token_ids
            }
            can_stop = _number_done(state)

            if not candidates and not can_stop:
                # Nothing legal to write and the number is not
                # complete either -- give up rather than spin.
                break

            logits = self.model.get_logits_from_input_ids(input_ids)

            best_token: Optional[int] = None
            best_logit: Optional[float] = None

            for token_id, ch in candidates.items():
                logit = logits[token_id]
                if best_token is None or logit > best_logit:
                    best_token, best_logit = token_id, logit

            if can_stop:
                stop_logit = logits[closer_first_id]
                if best_token is None or stop_logit >= best_logit:
                    # The model prefers what comes after the value
                    # over writing one more digit: the number is done.
                    break

            assert best_token is not None  # candidates was non-empty

            text += candidates[best_token]
            input_ids = input_ids + [best_token]
            state = _number_step(state, candidates[best_token]) or state

        value: Optional[Union[int, float]] = None
        if text and _number_done(state):
            value = int(text) if integer_only else float(text)

        input_ids = input_ids + self._encode_literal(closer_text)

        return value, input_ids

    def _string_safe_ids(self) -> np.ndarray:
        if self._string_safe_ids_cache is None:
            self._string_safe_ids_cache = self._load_string_safe_ids()
        return self._string_safe_ids_cache

    def _load_string_safe_ids(self) -> np.ndarray:
        """
        Build, once, the set of token ids that may appear inside the
        *unescaped* body of a JSON string: read the tokenizer's own
        vocabulary file, undo the leading-space marker byte-level BPE
        writes as "Ġ", and refuse any token whose text still contains
        the closing quote, a backslash, or a raw control character.

        Both excluded characters have their own dedicated handling
        instead: a backslash is offered separately as the single
        token that opens an escape sequence (see
        _generate_string_value), and a bare quote is never body
        content because it is what closes the string.
        """
        vocab_path = self.model.get_path_to_vocab_file()

        with open(vocab_path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)

        safe_ids = []

        for token_text, token_id in vocab.items():
            spelled = token_text.replace("Ġ", " ")

            if any(ch in _FORBIDDEN_STRING_CHARS for ch in spelled):
                continue
            if any(ord(ch) < 0x20 for ch in spelled):
                continue
            if any(ord(ch) > 0x7E and ch != " " for ch in spelled):
                continue

            safe_ids.append(token_id)

        return np.array(sorted(safe_ids), dtype=np.int64)

    def _generate_string_value(
        self,
        input_ids: List[int],
        closer_text: str,
        max_chars: int = 100,
    ) -> Tuple[str, List[int]]:
        safe_ids = self._string_safe_ids()
        text = ""
        state = "body"
        
        # Храним ID токенов, которые уже сгенерированы внутри этой строки
        recent_token_ids: List[int] = []

        while len(text) < max_chars:
            logits = np.asarray(
                self.model.get_logits_from_input_ids(input_ids)
            )

            if state == "escape":
                best_id, ch = self._best_of(logits, self._escape_token_ids)
                if best_id is None or ch is None:
                    break
                text += _ESCAPE_MAP[ch]
                input_ids = input_ids + [best_id]
                recent_token_ids.append(best_id)
                state = "body"
                continue

            # Штрафуем то, что уже недавно использовалось (простой штраф за повторение)
            for tid in set(recent_token_ids[-15:]):
                if 0 <= tid < len(logits):
                    logits[tid] -= 2.0  # Снижаем шанс повторного выбора того же токена

            best_content_id, best_content_logit = self._best_masked(
                logits, safe_ids
            )
            options: List[Tuple[float, str, Optional[int]]] = [
                (best_content_logit, "content", best_content_id),
            ]
            quote_id = self._quote_id
            if quote_id is not None:
                options.append((float(logits[quote_id]), "stop", None))
            backslash_id = self._backslash_id
            if backslash_id is not None:
                options.append(
                    (float(logits[backslash_id]), "escape", backslash_id)
                )

            _, action, token_id = max(options, key=lambda option: option[0])

            if action == "stop":
                break

            if action == "escape":
                assert token_id is not None
                input_ids = input_ids + [token_id]
                recent_token_ids.append(token_id)
                state = "escape"
                continue

            assert token_id is not None
            piece = self.model.decode([token_id])
            text += piece
            input_ids = input_ids + [token_id]
            recent_token_ids.append(token_id)

        input_ids = input_ids + self._encode_literal(closer_text)
        return text, input_ids

    @staticmethod
    def _best_masked(
        logits: np.ndarray,
        ids: np.ndarray,
    ) -> Tuple[int, float]:
        """Argmax of logits, restricted to a numpy array of ids."""
        masked = logits[ids]
        index = int(np.argmax(masked))
        return int(ids[index]), float(masked[index])

    @staticmethod
    def _best_of(
        logits: np.ndarray,
        candidates: Dict[int, str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Argmax of logits, restricted to a small {id: label} map."""
        best_id: Optional[int] = None
        best_label: Optional[str] = None
        best_logit: Optional[float] = None
        for token_id, label in candidates.items():
            value = float(logits[token_id])
            if best_logit is None or value > best_logit:
                best_id, best_label, best_logit = token_id, label, value
        return best_id, best_label
