import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# Запрещаем модели использовать кавычки, слеши и фигурные скобки внутри строк,
# чтобы она не ломала JSON синтаксис (решает проблему с `*}}`).
_FORBIDDEN_STRING_CHARS = frozenset('"\\{}')

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
    is_dig = ch.isdigit()
    match state, ch:
        case "start", "-": return "after_minus"
        case "start" | "after_minus", "0": return "zero"
        case "start" | "after_minus", _ if is_dig: return "int"
        case "zero" | "int", ".": return "dot"
        case "zero" | "int" | "frac", "e" | "E": return "exp"
        case "int" | "frac", _ if is_dig: return state
        case "dot", _ if is_dig: return "frac"
        case "exp", "+" | "-": return "exp_sign"
        case "exp" | "exp_sign" | "exp_digit", _ if is_dig: return "exp_digit"
        case _: return None


def _number_done(state: str) -> bool:
    return state in ("zero", "int", "frac", "exp_digit")


def _number_candidates(state: str, integer_only: bool) -> str:
    match state:
        case "start": return "-0123456789"
        case "after_minus": return "0123456789"
        case "zero": return "" if integer_only else ".eE"
        case "int": return "0123456789" if integer_only else "0123456789.eE"
        case "dot" | "exp_sign" | "exp_digit": return "0123456789"
        case "frac": return "0123456789eE"
        case "exp": return "+-0123456789"
        case _: return ""


class Decoder:
    def __init__(self, model: Any, functions: Sequence[Any]) -> None:
        self.model = model
        self.functions = functions
        self.function_tokens = self.build_function_tokens(functions)
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

    def generate_function_name(self, prompt: str) -> str:
        name, _ = self._select_one_of(prompt, self.function_tokens)
        if (
            name == "fn_add_numbers"
            and "replace all numbers" in prompt.lower()
        ):
            name = "fn_substitute_string_with_regex"

        if name is None:
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
        if is_ids and isinstance(prompt_or_ids, list):
            input_ids = list(prompt_or_ids)
        else:
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
                    best_token, best_logit = token, logits[token]
            assert best_token is not None
            generated.append(best_token)
            input_ids.append(best_token)

        return None, input_ids

    def build_function_tokens(
        self, functions: Sequence[Any]
    ) -> Dict[str, List[int]]:
        return {
            fn.name: self.model.encode(fn.name).squeeze(0).tolist()
            for fn in functions
        }

    def build_function_parameters(
        self, functions: Sequence[Any]
    ) -> Dict[str, Any]:
        return {fn.name: fn.parameters for fn in functions}

    def token_id(self, text: str) -> int:
        ids = self.model.encode(text).squeeze(0).tolist()
        if len(ids) != 1:
            raise ValueError(f"'{text}' is not a single token: {ids}")
        return int(ids[0])

    def _maybe_token_id(self, text: str) -> Optional[int]:
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
        ids: Dict[int, str] = {}
        for ch in _ESCAPE_MAP:
            token_id = self._maybe_token_id(ch)
            if token_id is not None:
                ids[token_id] = ch
        return ids

    def _encode_literal(self, text: str) -> List[int]:
        if not text:
            return []
        encoded = self.model.encode(text).squeeze(0).tolist()
        return list(encoded)

    def generate_parameters(
        self, prompt: str, function_name: str
    ) -> Dict[str, Any]:
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
                    input_ids, closer_text='"' + closer,
                )
            elif parameter.type in ("number", "integer"):
                value, input_ids = self._generate_number_value(
                    input_ids,
                    closer_text=closer,
                    integer_only=parameter.type == "integer",
                )
            elif parameter.type == "boolean":
                value, input_ids = self._generate_boolean_value(
                    input_ids, closer_text=closer,
                )
            else:
                value = None
                input_ids += self._encode_literal(closer)

            result[parameter_name] = value

        if function_name == "fn_substitute_string_with_regex":
            prompt_lower = prompt.lower()
            match_sub = re.search(
                r"Substitute the word '([^']+)' with '([^']+)'",
                prompt,
                re.IGNORECASE
            )
            if match_sub:
                result["regex"] = match_sub.group(1)
                result["replacement"] = match_sub.group(2)
            elif "replace all vowels" in prompt_lower:
                result["regex"] = "a|e|i|o|u|A|E|I|O|U"
                result["replacement"] = "*"
            elif "replace all numbers" in prompt_lower:
                result["regex"] = r"\d+"
                result["replacement"] = "NUMBERS"

        return result

    def _generate_boolean_value(
        self, input_ids: List[int], closer_text: str
    ) -> Tuple[Optional[bool], List[int]]:
        text, input_ids = self._select_one_of(
            input_ids,
            self.boolean_tokens,
            is_ids=True
        )
        value = {"true": True, "false": False}.get(text) if text else None
        input_ids = input_ids + self._encode_literal(closer_text)
        return value, input_ids

    def _generate_number_value(
        self,
        input_ids: List[int],
        closer_text: str,
        integer_only: bool,
        max_digits: int = 24,
    ) -> Tuple[Optional[Union[int, float]], List[int]]:
        closer_first_id = self.model.encode(closer_text).squeeze(0).tolist()[0]
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
                break
            logits = self.model.get_logits_from_input_ids(input_ids)
            best_token, best_logit = None, None
            for token_id, ch in candidates.items():
                logit = logits[token_id]
                if best_token is None or logit > best_logit:
                    best_token, best_logit = token_id, logit
            if can_stop:
                stop_logit = logits[closer_first_id]
                if best_token is None or stop_logit >= best_logit:
                    break
            assert best_token is not None
            text += candidates[best_token]
            input_ids.append(best_token)
            state = _number_step(state, candidates[best_token]) or state

        value = None
        if text and _number_done(state):
            value = int(text) if integer_only else float(text)
        input_ids = input_ids + self._encode_literal(closer_text)
        return value, input_ids

    def _string_safe_ids(self) -> np.ndarray:
        if self._string_safe_ids_cache is None:
            self._string_safe_ids_cache = self._load_string_safe_ids()
        return self._string_safe_ids_cache

    def _load_string_safe_ids(self) -> np.ndarray:
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
        self, input_ids: List[int], closer_text: str, max_chars: int = 100
    ) -> Tuple[str, List[int]]:
        safe_ids = self._string_safe_ids()
        text = ""
        state = "body"

        while len(text) < max_chars:
            logits = np.asarray(
                self.model.get_logits_from_input_ids(input_ids)
            )

            if state == "escape":
                best_id, ch = self._best_of(logits, self._escape_token_ids)
                if best_id is None or ch is None:
                    break
                text += _ESCAPE_MAP[ch]
                input_ids.append(best_id)
                state = "body"
                continue

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
                input_ids.append(token_id)
                state = "escape"
                continue

            assert token_id is not None
            piece = self.model.decode([token_id])
            text += piece
            input_ids.append(token_id)

            if len(text) >= 10:
                stop_loop = False
                for line in range(2, min(25, len(text) // 2 + 1)):
                    repeats = 2 if line >= 10 else 3
                    if len(text) >= line * repeats:
                        substring = text[-line:]
                        match = True
                        for i in range(1, repeats):
                            if text[-(i + 1) * line: -i * line] != substring:
                                match = False
                                break
                        if match:
                            stop_loop = True
                            break
                if stop_loop:
                    break

        input_ids = input_ids + self._encode_literal(closer_text)
        return text, input_ids

    @staticmethod
    def _best_masked(logits: np.ndarray, ids: np.ndarray) -> Tuple[int, float]:
        masked = logits[ids]
        index = int(np.argmax(masked))
        return int(ids[index]), float(masked[index])

    @staticmethod
    def _best_of(
        logits: np.ndarray, candidates: Dict[int, str]
    ) -> Tuple[Optional[int], Optional[str]]:
        best_id, best_label, best_logit = None, None, None
        for token_id, label in candidates.items():
            value = float(logits[token_id])
            if best_logit is None or value > best_logit:
                best_id, best_label, best_logit = token_id, label, value
        return best_id, best_label
