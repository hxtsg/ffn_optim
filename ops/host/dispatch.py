"""Implementation selection: no silent fallbacks, no device imports."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CATLASS = ROOT / "3rdparty/catlass"
PINNED_COMMIT = "769cd40a8716b28650b6bebb08db4834eea4462f"
EXAMPLES = CATLASS / "python/tla_dsl/examples/end_to_end"
SOURCES = {
    "basic": EXAMPLES / "basic_mmad/basic_matmul.py",
    "streamk": EXAMPLES / "basic_mmad_streamk/basic_mmad_streamk.py",
}
DOWN_IMPLS = ("basic", "full_load_a", "full_load_b", "streamk")


class DependencyUnavailable(RuntimeError):
    pass


class NotApplicable(ValueError):
    pass


@dataclass(frozen=True)
class Selection:
    up_impl: str = "basic"
    down_impl: str = "basic"

    def __post_init__(self):
        if self.up_impl != "basic":
            raise ValueError("V1 up_impl must be basic")
        if self.down_impl not in DOWN_IMPLS:
            raise ValueError(f"Unknown down_impl: {self.down_impl}")

    def require_available(self):
        if self.down_impl in ("full_load_a", "full_load_b"):
            raise DependencyUnavailable(
                f"{self.down_impl}: no reusable DSL entry identified at CATLASS v2.0.0; no fallback"
            )
        return self


def verify_dependency():
    """Check the fixed checkout and tracked modifications, without importing DSL."""
    if not (CATLASS / ".git").exists():
        raise DependencyUnavailable("Initialize 3rdparty/catlass submodule first")
    head = subprocess.run(["git", "-C", str(CATLASS), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()
    if head != PINNED_COMMIT:
        raise DependencyUnavailable(f"CATLASS commit mismatch: {head}")
    changed = subprocess.run(["git", "-C", str(CATLASS), "status", "--porcelain",
                              "--untracked-files=no"], check=True, text=True,
                             capture_output=True).stdout
    if changed.strip():
        raise DependencyUnavailable("Tracked CATLASS files are modified; restore or review the dependency")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in SOURCES.items()}
