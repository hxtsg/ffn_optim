"""Conservative adapters to CATLASS example TilingParams (not baseline autotiling)."""
from dataclasses import dataclass, asdict
from .inputs import Problem
from .dispatch import Selection, NotApplicable


def align(value, unit):
    return (value + unit - 1) // unit * unit


@dataclass(frozen=True)
class Tile:
    """功能：保存单段MatMul的L1/L0分块参数，尺寸单位为元素数

    输入：l1_tm/tn/tk和l0_tm/tn/tk为各层M/N/K块大小，须满足底层样例约束
    当前make_plan固定使用L1=(64,64,128)、L0=(64,64,32)，不支持任意调参承诺
    输出：只读分块配置，编译时转换为CATLASS的TilingParams；本类不自行校验
    """

    l1_tm: int = 64
    l1_tn: int = 64
    l1_tk: int = 128
    l0_tm: int = 64
    l0_tn: int = 64
    l0_tk: int = 32


@dataclass(frozen=True)
class Plan:
    """功能：汇总一次FFN执行所需的形状、策略、分块及workspace计划

    输入：已校验的problem/selection、统一block_num、补零后的mp/kp/hp/np
    up/down为Tile，workspace_shape为二维FP32缓冲形状，应由make_plan生成
    执行前还须核验block_num不超过物理AIC数，直接构造本类不会执行这些检查
    输出：workspace_bytes返回scratch字节数，不含hidden和其他缓冲
    to_dict()返回可序列化的嵌套字典；本类不分配内存或下发Kernel
    """

    problem: Problem
    selection: Selection
    block_num: int
    mp: int
    kp: int
    hp: int
    np: int
    up: Tile
    down: Tile
    workspace_shape: tuple[int, int]

    @property
    def workspace_bytes(self):
        return self.workspace_shape[0] * self.workspace_shape[1] * 4

    def to_dict(self):
        return asdict(self)


def make_plan(problem, selection=Selection(), block_num=8):
    selection.require_available()
    if type(block_num) is not int or not 1 <= block_num <= 65535:
        raise ValueError("block_num must be a positive integer; hardware limit checked before launch")
    tile = Tile()
    mp, kp = align(problem.m, 64), align(problem.k, 16)
    hp, np = align(problem.h, 64), align(problem.n, 64)
    if selection.down_impl == "streamk":
        tiles = (mp // 64) * (np // 64)
        if tiles % block_num == 0:
            raise NotApplicable("Stream-K candidate has no remainder tiles for this block_num")
        if (hp + tile.l1_tk - 1) // tile.l1_tk < 2:
            raise NotApplicable("Stream-K candidate requires at least two L1 K tiles")
        workspace = (tile.l1_tm * 2 * block_num, tile.l1_tn)
    else:
        workspace = (1, 1)  # unused argument, never read in the Basic variant
    return Plan(problem, selection, block_num, mp, kp, hp, np, tile, tile, workspace)
