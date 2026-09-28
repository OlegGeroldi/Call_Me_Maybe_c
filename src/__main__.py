import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from .decoder import Decoder, build_context
from .structures import (
    FunctionCallingTest,
    FunctionCallResult,
    FunctionDefinition,
)

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FUNCTIONS = ROOT / "data" / "input" / "functions_definition.json"
DEFAULT_INPUT = ROOT / "data" / "input" / "function_calling_tests.json"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "function_calling_results.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="src")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def load_functions(path: Path) -> List[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [FunctionDefinition.model_validate(item) for item in raw]


def load_tests(path: Path) -> List[FunctionCallingTest]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [FunctionCallingTest.model_validate(item) for item in raw]


def format_elapsed(seconds: float) -> str:
    """Render a duration as e.g. "1m 07s"."""
    minutes, whole_seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m {whole_seconds:02d}s"


def main(argv: Optional[Sequence[str]] = None) -> int:

    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        functions = load_functions(args.functions_definition)
        tests = load_tests(args.input)

    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}", file=sys.stderr)
        return 1

    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 1

    except ValidationError as e:
        print("Validation error:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    try:
        llm = Small_LLM_Model()
    except Exception as e:  # noqa: BLE001 -- reported, never a traceback
        print(f"Error: could not load the model: {e}", file=sys.stderr)
        return 1

    decoder = Decoder(llm, functions)
    context = build_context(functions)

    results = []
    failed = 0

    for index, test in enumerate(tests, start=1):

        user_prompt = test.prompt
        started = time.perf_counter()

        prompt = (
            f"{context}\n"
            f"User request:\n"
            f"{user_prompt}\n\n"
            f"The best matching function is:\n"
        )
        try:
            model_output = decoder.generate_function_name(prompt)
            parameters = decoder.generate_parameters(prompt, model_output)

        except Exception as e:  # noqa: BLE001 -- one bad prompt must
            # not lose every answer already produced by the others.
            elapsed = format_elapsed(time.perf_counter() - started)
            print(
                f"[{index}/{len(tests)}] {elapsed} - FAILED "
                f"{user_prompt!r}: {e}",
                file=sys.stderr,
            )
            failed += 1
            continue

        elapsed = format_elapsed(time.perf_counter() - started)
        print(
            f"[{index}/{len(tests)}] {elapsed} - {user_prompt!r} "
            f"-> {model_output}({parameters})"
        )

        result = FunctionCallResult(
            prompt=user_prompt,
            name=model_output,
            parameters=parameters,
        )

        results.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(results)} results to {args.output}")
    if failed:
        print(f"{failed} prompt(s) could not be answered", file=sys.stderr)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
