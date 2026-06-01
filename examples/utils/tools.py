import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileTools:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root).resolve()

    def list_files(self, pattern: str = "*.md") -> list[str]:
        """List files matching a glob pattern."""
        return sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern))

    def grep(self, pattern: str, max_results: int = 30) -> list[str]:
        """Search markdown files for lines matching `pattern`. Returns 'file:lineno: text'."""
        rx = re.compile(pattern, re.IGNORECASE)
        hits: list[str] = []
        for f in sorted(self.root.rglob("*.md")):
            for i, line in enumerate(f.read_text().splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()}")
                    if len(hits) >= max_results:
                        return hits
        return hits

    def read_file(self, path: str) -> str:
        """Read a file by relative path. Refuses paths outside the scoped directory."""
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Path '{path}' is outside the allowed directory")
        return target.read_text()

    @property
    def tools(self) -> list:
        """Return bound methods ready for LLM tool-calling."""
        return [self.list_files, self.grep, self.read_file]
