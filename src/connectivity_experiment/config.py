from __future__ import annotations

from pathlib import Path


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
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            child: dict = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def repo_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[1]
