"""Conservative adapters to CATLASS example TilingParams (not baseline autotiling)."""
from dataclasses import dataclass, asdict
from .inputs import Problem
from .dispatch import Selection, NotApplicable


def align(value, unit):
    return (value + unit - 1) // unit * unit


@dataclass(frozen=True)
class Tile:
    l1_tm: int = 64
    l1_tn: int = 64
    l1_tk: int = 128
    l0_tm: int = 64
    l0_tn: int = 64
    l0_tk: int = 32


@dataclass(frozen=True)
class Plan:
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
