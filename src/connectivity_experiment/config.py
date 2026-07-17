from __future__ import annotations

from pathlib import Path


def _iter_config_lines(lines: list[str]) -> list[tuple[int, str, str]]:
    parsed_lines: list[tuple[int, str, str]] = []
    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        parsed_lines.append((indent, stripped, raw_line))
    return parsed_lines


def _next_is_list(items: list[tuple[int, str, str]], start_index: int, parent_indent: int) -> bool:
    for index in range(start_index + 1, len(items)):
        indent, stripped, _ = items[index]
        if indent <= parent_indent:
            return False
        return stripped.startswith("- ")
    return False


def _strip_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    result = []
    for char in line:
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
                quote_char = ""
        if char == "#" and not in_quote:
            break
        result.append(char)
    return "".join(result).rstrip()


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict:
    path = Path(path)
    items = _iter_config_lines(path.read_text(encoding="utf-8").splitlines())
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for index, (indent, stripped, raw_line) in enumerate(items):
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(current, list):
                raise ValueError(f"Invalid list item outside of list context: {raw_line}")
            current.append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(current, dict):
            raise ValueError(f"Invalid mapping entry inside list context: {raw_line}")
        if not value:
            child = [] if _next_is_list(items, index, indent) else {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def repo_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[1]
