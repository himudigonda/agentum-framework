from pathlib import Path

from agentum import tool


@tool
def write_file(filepath: str, content: str) -> str:
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"


@tool
def read_file(filepath: str) -> str:
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found at {filepath}"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {filepath}: {e}"
