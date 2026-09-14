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
KERNEL_ENTRIES = {
    "basic": "ffn_basic_gelu_basic_kernel",
    "streamk": "ffn_basic_gelu_streamk_kernel",
}


class DependencyUnavailable(RuntimeError):
    """功能：区分依赖缺失、版本不符和候选实现尚未接入的错误

    输入：调用方提供描述缺失原因的错误消息字符串
    输出：可捕获的RuntimeError异常，str(exc)返回原因，不触发策略回退
    """

    pass


class NotApplicable(ValueError):
    """功能：表示已有候选实现不适用于当前shape或核数配置

    输入：调用方提供具体适用性限制的错误消息字符串
    输出：可捕获的ValueError异常，供搜索记录not_applicable，不执行替代路径
    """

    pass


@dataclass(frozen=True)
class Selection:
    """功能：指定单次FFN的上、下投影实现，不自动选择或切换策略

    输入：up_impl只能为basic；down_impl可为basic、streamk、full_load_a或full_load_b
    输出：合法名称生成只读配置，非法名称在构造时抛出ValueError
    require_available()对Basic/Stream-K返回自身，对未接入的全载抛出DependencyUnavailable
    候选的shape及核数适用性由make_plan另行检查
    """

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


def read_kernel_source(down_impl="basic"):
    """读取普通Kernel源文件用于审查和快照，不导入DSL、不生成或改写代码"""
    Selection(down_impl=down_impl).require_available()
    hashes = verify_dependency()
    source = (ROOT / "ops/kernel/ffn.py").read_text()
    return source, {"donor_sha256": hashes,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "down_impl": down_impl, "entry_point": KERNEL_ENTRIES[down_impl],
                    "source_file": "ops/kernel/ffn.py"}


def compile_kernel(tensors, plan):
    """显式设备编译入口：输入六个DSL Tensor及Plan，返回所选函数的artifact和来源信息

    仅在设备路径导入DSL模块，两条路径只编译其中一条，不在此处下发Kernel
    """
    from dataclasses import asdict

    _, provenance = read_kernel_source(plan.selection.down_impl)
    import catlass.tla as tla
    from ops.kernel import ffn as kernels

    kernel = getattr(kernels, provenance["entry_point"])
    artifact = tla.compile(
        kernel, *tensors,
        kernels.TilingParams(**asdict(plan.up)),
        kernels.TilingParams(**asdict(plan.down)), kernels.SwizzleParams(),
        plan.block_num, tla.params.HF32Mode.HF32_DISABLE,
        options="--npu-arch 3510",
    )
    return artifact, provenance
