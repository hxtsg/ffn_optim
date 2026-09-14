# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""两种直接编写的单次下发FFN Kernel，Host只编译所选函数

MatMul供数、分块和Stream-K归约沿用CATLASS v2.0.0样例
GELU、UB握手和阶段同步在每个函数内展开，不使用源码字符串或AST拼装
本模块仅在明确进入设备编译路径时导入
"""
from dataclasses import dataclass
import catlass.tla as tla


@dataclass(frozen=True)
class TilingParams:
    """功能：向Kernel传递与CATLASS样例一致的编译期分块参数

    输入：由Host的Tile转换，当前L1=(64,64,128)、L0=(64,64,32)
    输出：六个Constexpr字段，不执行计算或自行校验资源容量
    """

    l1_tm: tla.Constexpr[int] = 64
    l1_tn: tla.Constexpr[int] = 64
    l1_tk: tla.Constexpr[int] = 128
    l0_tm: tla.Constexpr[int] = 64
    l0_tn: tla.Constexpr[int] = 64
    l0_tk: tla.Constexpr[int] = 32


@dataclass(frozen=True)
class SwizzleParams:
    """功能：向Stream-K调度传递编译期Swizzle参数

    输入：方向默认为0，偏移默认为3，沿用CATLASS样例参数
    输出：两个Constexpr字段，由Stream-K计算tile访问顺序；Basic路径不使用
    """

    SWIZZLE_DIRECTION: tla.Constexpr[int] = 0
    SWIZZLE_OFFSET: tla.Constexpr[int] = 3


AIV_TILE_M = 16
AIV_SUB_BLOCK_NUM = 2
AIV_REG_M = 1
AIV_REG_N = 64


@tla.kernel
def ffn_basic_gelu_basic_kernel(
    x: tla.Tensor,
    weight1: tla.Tensor,
    weight2: tla.Tensor,
    hidden: tla.Tensor,
    output: tla.Tensor,
    workspace: tla.Tensor,
    up_tiling: TilingParams,
    down_tiling: TilingParams,
    swizzle: SwizzleParams,
    block_dim: tla.Constexpr[int],
    hf32_mode: tla.Constexpr[tla.params.HF32Mode],
):
    """Basic上投影→GELU→Basic下投影，单Kernel完成

    输入：同设备FP16/BF16的补零X/W1/W2，hidden/output为同dtype缓冲，workspace为FP32
    逻辑形状为X[mp,kp]、W1[kp,hp]、W2[hp,np]，权重底层采用Linear转置存储
    mp/hp/np按64补齐，kp按16补齐，Host检查分块、核数和路径适用性
    输出：写hidden[mp,hp]和output[mp,np]，无Python返回值，Host负责裁剪恢复形状
    """
    up_c0 = 0
    up_c1 = 1
    up_dtype_a = x.ptr.dtype
    up_dtype_b = weight1.ptr.dtype
    up_m = x.origin_shape[0]
    up_n = weight1.origin_shape[1]
    up_k = x.origin_shape[1]
    up_l1a0_data_ready = tla.flag('up_l1a0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1a1_data_ready = tla.flag('up_l1a1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1b0_data_ready = tla.flag('up_l1b0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1b1_data_ready = tla.flag('up_l1b1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1a0_available = tla.flag('up_l1a0_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1a1_available = tla.flag('up_l1a1_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1b0_available = tla.flag('up_l1b0_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1b1_available = tla.flag('up_l1b1_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l0a0_available = tla.flag('up_l0a0_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0a1_available = tla.flag('up_l0a1_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0b0_available = tla.flag('up_l0b0_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0b1_available = tla.flag('up_l0b1_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0_ab_data_ready = tla.flag('up_l0_ab_data_ready', tla.arch.MTE1, tla.arch.CUBE)
    up_l0c_available = tla.flag('up_l0c_available', tla.arch.FIX, tla.arch.CUBE)
    up_l1a0_ptr = tla.allocate(up_tiling.l1_tm * up_tiling.l1_tk, up_dtype_a, tla.AddressSpace.l1, 512)
    up_l1a1_ptr = tla.allocate(up_tiling.l1_tm * up_tiling.l1_tk, up_dtype_a, tla.AddressSpace.l1, 512)
    up_l1b0_ptr = tla.allocate(up_tiling.l1_tk * up_tiling.l1_tn, up_dtype_b, tla.AddressSpace.l1, 512)
    up_l1b1_ptr = tla.allocate(up_tiling.l1_tk * up_tiling.l1_tn, up_dtype_b, tla.AddressSpace.l1, 512)
    up_l0a0_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tk, up_dtype_a, tla.AddressSpace.l0a, 512)
    up_l0a1_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tk, up_dtype_a, tla.AddressSpace.l0a, 512)
    up_l0b0_ptr = tla.allocate(up_tiling.l0_tk * up_tiling.l0_tn, up_dtype_b, tla.AddressSpace.l0b, 512)
    up_l0b1_ptr = tla.allocate(up_tiling.l0_tk * up_tiling.l0_tn, up_dtype_b, tla.AddressSpace.l0b, 512)
    up_l0c_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 512)
    up_grid_m = (up_m + up_tiling.l1_tm - 1) // up_tiling.l1_tm
    up_grid_n = (up_n + up_tiling.l1_tn - 1) // up_tiling.l1_tn
    up_total_blocks = up_grid_m * up_grid_n
    down_c0 = 0
    down_c1 = 1
    down_dtype_a = hidden.ptr.dtype
    down_dtype_b = weight2.ptr.dtype
    down_m = hidden.origin_shape[0]
    down_n = weight2.origin_shape[1]
    down_k = hidden.origin_shape[1]
    down_l1a0_data_ready = tla.flag('down_l1a0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1a1_data_ready = tla.flag('down_l1a1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1b0_data_ready = tla.flag('down_l1b0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1b1_data_ready = tla.flag('down_l1b1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1a0_available = tla.flag('down_l1a0_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1a1_available = tla.flag('down_l1a1_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1b0_available = tla.flag('down_l1b0_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1b1_available = tla.flag('down_l1b1_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l0a0_available = tla.flag('down_l0a0_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0a1_available = tla.flag('down_l0a1_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0b0_available = tla.flag('down_l0b0_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0b1_available = tla.flag('down_l0b1_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0_ab_data_ready = tla.flag('down_l0_ab_data_ready', tla.arch.MTE1, tla.arch.CUBE)
    down_l0c_available = tla.flag('down_l0c_available', tla.arch.FIX, tla.arch.CUBE)
    down_l1a0_ptr = tla.allocate(
        down_tiling.l1_tm * down_tiling.l1_tk,
        down_dtype_a,
        tla.AddressSpace.l1,
        512,
    )
    down_l1a1_ptr = tla.allocate(
        down_tiling.l1_tm * down_tiling.l1_tk,
        down_dtype_a,
        tla.AddressSpace.l1,
        512,
    )
    down_l1b0_ptr = tla.allocate(
        down_tiling.l1_tk * down_tiling.l1_tn,
        down_dtype_b,
        tla.AddressSpace.l1,
        512,
    )
    down_l1b1_ptr = tla.allocate(
        down_tiling.l1_tk * down_tiling.l1_tn,
        down_dtype_b,
        tla.AddressSpace.l1,
        512,
    )
    down_l0a0_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tk,
        down_dtype_a,
        tla.AddressSpace.l0a,
        512,
    )
    down_l0a1_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tk,
        down_dtype_a,
        tla.AddressSpace.l0a,
        512,
    )
    down_l0b0_ptr = tla.allocate(
        down_tiling.l0_tk * down_tiling.l0_tn,
        down_dtype_b,
        tla.AddressSpace.l0b,
        512,
    )
    down_l0b1_ptr = tla.allocate(
        down_tiling.l0_tk * down_tiling.l0_tn,
        down_dtype_b,
        tla.AddressSpace.l0b,
        512,
    )
    down_l0c_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tn,
        tla.Float32,
        tla.AddressSpace.l0c,
        512,
    )
    down_grid_m = (down_m + down_tiling.l1_tm - 1) // down_tiling.l1_tm
    down_grid_n = (down_n + down_tiling.l1_tn - 1) // down_tiling.l1_tn
    down_total_blocks = down_grid_m * down_grid_n
    tile_available = tla.cross_flag('ffn_tile_available', mode=4)
    tile_ready = tla.cross_flag('ffn_tile_ready', mode=4)
    up_ready = tla.cross_flag('ffn_up_ready', mode=2)
    all_aic = tla.cross_flag('ffn_all_aic', mode=0)
    down_release = tla.cross_flag('ffn_down_release', mode=2)
    gelu_done = tla.flag('ffn_gelu_done', tla.arch.VECTOR, tla.arch.MTE3)
    acc_ptr = tla.allocate(4096, tla.Float32, tla.AddressSpace.ub, 256)
    out_ptr = tla.allocate(4096, hidden.ptr.dtype, tla.AddressSpace.ub, 256)
    ub_layout = tla.make_layout(tla.make_shape(64, 64), tla.make_stride(64, 1), layoutTag=tla.arch.RowMajor)
    acc_ub = tla.make_tensor(acc_ptr, ub_layout)
    out_ub = tla.make_tensor(out_ptr, ub_layout)
    acc_1d = tla.make_tensor(acc_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
    out_1d = tla.make_tensor(out_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
    cast_params = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    store_params = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
    # AIC：上投影、L0C→UB、阶段屏障、下投影
    with tla.cube():
        tla.set_flag(up_l1a0_available)
        tla.set_flag(up_l1a1_available)
        tla.set_flag(up_l1b0_available)
        tla.set_flag(up_l1b1_available)
        tla.set_flag(up_l0a0_available)
        tla.set_flag(up_l0a1_available)
        tla.set_flag(up_l0b0_available)
        tla.set_flag(up_l0b1_available)
        tla.set_flag(up_l0c_available)
        up_runtime_zero = tla.as_numeric(0)
        up_l1_buf_idx = up_runtime_zero
        up_l0_buf_idx = up_runtime_zero
        up_block_range = tla.range(tla.arch.block_idx(), up_total_blocks, tla.arch.block_num())
        for up_block_linear in up_block_range:
            up_block_row = up_block_linear // up_grid_n
            up_block_col = up_block_linear % up_grid_n
            up_gm_a_by_core = tla.tile_view(
                x,
                tla.make_shape(up_tiling.l1_tm, up_k),
                tla.make_coord(up_block_row, up_c0),
            )
            up_gm_b_by_core = tla.tile_view(
                weight1,
                tla.make_shape(up_k, up_tiling.l1_tn),
                tla.make_coord(up_c0, up_block_col),
            )
            up_gm_c_by_core = tla.tile_view(
                hidden,
                tla.make_shape(up_tiling.l1_tm, up_tiling.l1_tn),
                tla.make_coord(up_block_row, up_block_col),
            )
            up_k_block = up_gm_a_by_core.origin_shape[1]
            up_k_l1_count = (up_k_block + up_tiling.l1_tk - 1) // up_tiling.l1_tk
            up_k_l1_range = tla.range(up_c0, up_k_l1_count, up_c1)
            up_l0_c = tla.make_tensor_like(up_l0c_ptr, up_gm_c_by_core)
            for up_k_l1 in up_k_l1_range:
                up_gm_a_by_l1 = tla.tile_view(
                    up_gm_a_by_core,
                    tla.make_shape(up_tiling.l1_tm, up_tiling.l1_tk),
                    tla.make_coord(up_c0, up_k_l1),
                )
                up_gm_b_by_l1 = tla.tile_view(
                    up_gm_b_by_core,
                    tla.make_shape(up_tiling.l1_tk, up_tiling.l1_tn),
                    tla.make_coord(up_k_l1, up_c0),
                )
                up_l1_a = tla.make_tensor_like(
                    up_l1a0_ptr if up_l1_buf_idx == up_c0 else up_l1a1_ptr,
                    up_gm_a_by_l1,
                )
                up_l1_b = tla.make_tensor_like(
                    up_l1b0_ptr if up_l1_buf_idx == up_c0 else up_l1b1_ptr,
                    up_gm_b_by_l1,
                )
                if up_l1_buf_idx == up_c0:
                    tla.wait_flag(up_l1a0_available)
                else:
                    tla.wait_flag(up_l1a1_available)
                tla.copy(up_l1_a, up_gm_a_by_l1)
                if up_l1_buf_idx == up_c0:
                    tla.set_flag(up_l1a0_data_ready)
                else:
                    tla.set_flag(up_l1a1_data_ready)
                if up_l1_buf_idx == up_c0:
                    tla.wait_flag(up_l1b0_available)
                else:
                    tla.wait_flag(up_l1b1_available)
                tla.copy(up_l1_b, up_gm_b_by_l1)
                if up_l1_buf_idx == up_c0:
                    tla.set_flag(up_l1b0_data_ready)
                else:
                    tla.set_flag(up_l1b1_data_ready)
                up_k_l0_count = (up_l1_a.origin_shape[1] + up_tiling.l0_tk - 1) // up_tiling.l0_tk
                up_k_l0_range = tla.range(up_c0, up_k_l0_count, up_c1)
                for up_k_l0 in up_k_l0_range:
                    up_l1_a_by_l0 = tla.tile_view(
                        up_l1_a,
                        tla.make_shape(up_tiling.l0_tm, up_tiling.l0_tk),
                        tla.make_coord(up_c0, up_k_l0),
                    )
                    up_l1_b_by_l0 = tla.tile_view(
                        up_l1_b,
                        tla.make_shape(up_tiling.l0_tk, up_tiling.l0_tn),
                        tla.make_coord(up_k_l0, up_c0),
                    )
                    up_l0_a = tla.make_tensor_like(
                        up_l0a0_ptr if up_l0_buf_idx == up_c0 else up_l0a1_ptr,
                        up_l1_a_by_l0,
                    )
                    up_l0_b = tla.make_tensor_like(
                        up_l0b0_ptr if up_l0_buf_idx == up_c0 else up_l0b1_ptr,
                        up_l1_b_by_l0,
                    )
                    if up_k_l0 == 0:
                        if up_l1_buf_idx == up_c0:
                            tla.wait_flag(up_l1a0_data_ready)
                        else:
                            tla.wait_flag(up_l1a1_data_ready)
                    if up_l0_buf_idx == up_c0:
                        tla.wait_flag(up_l0a0_available)
                    else:
                        tla.wait_flag(up_l0a1_available)
                    tla.copy(up_l0_a, up_l1_a_by_l0)
                    if up_k_l0 == up_k_l0_count - 1:
                        if up_l1_buf_idx == up_c0:
                            tla.set_flag(up_l1a0_available)
                        else:
                            tla.set_flag(up_l1a1_available)
                    if up_k_l0 == 0:
                        if up_l1_buf_idx == up_c0:
                            tla.wait_flag(up_l1b0_data_ready)
                        else:
                            tla.wait_flag(up_l1b1_data_ready)
                    if up_l0_buf_idx == up_c0:
                        tla.wait_flag(up_l0b0_available)
                    else:
                        tla.wait_flag(up_l0b1_available)
                    tla.copy(up_l0_b, up_l1_b_by_l0)
                    if up_k_l0 == up_k_l0_count - 1:
                        if up_l1_buf_idx == up_c0:
                            tla.set_flag(up_l1b0_available)
                        else:
                            tla.set_flag(up_l1b1_available)
                    tla.set_flag(up_l0_ab_data_ready)
                    tla.wait_flag(up_l0_ab_data_ready)
                    up_unit_flag = 3 if up_k_l1 == up_k_l1_count - 1 and up_k_l0 == up_k_l0_count - 1 else 2
                    up_init_c = True if up_k_l1 == 0 and up_k_l0 == 0 else False
                    if tla.const_expr(hf32_mode != tla.params.HF32Mode.HF32_DISABLE and up_dtype_a == tla.Float32 and (up_dtype_b == tla.Float32)):
                        tla.mmad(
                            up_l0_c,
                            up_l0_a,
                            up_l0_b,
                            init_c=up_init_c,
                            unit_flag=up_unit_flag,
                            hf32_mode=hf32_mode,
                        )
                    else:
                        tla.mmad(up_l0_c, up_l0_a, up_l0_b, init_c=up_init_c, unit_flag=up_unit_flag)
                    if up_l0_buf_idx == up_c0:
                        tla.set_flag(up_l0a0_available)
                        tla.set_flag(up_l0b0_available)
                    else:
                        tla.set_flag(up_l0a1_available)
                        tla.set_flag(up_l0b1_available)
                    up_l0_buf_idx = up_c1 - up_l0_buf_idx
                up_l1_buf_idx = up_c1 - up_l1_buf_idx
            tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
            tla.copy(
                acc_ub,
                up_l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0),
            )
            tla.cross_core_set_flag(tile_ready, tla.arch.FIX, aiv_id=0)
        tla.wait_flag(up_l1a0_available)
        tla.wait_flag(up_l1a1_available)
        tla.wait_flag(up_l1b0_available)
        tla.wait_flag(up_l1b1_available)
        tla.wait_flag(up_l0a0_available)
        tla.wait_flag(up_l0a1_available)
        tla.wait_flag(up_l0b0_available)
        tla.wait_flag(up_l0b1_available)
        tla.wait_flag(up_l0c_available)
        tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)
        # mode2报到→mode0全AIC汇合→mode2放行，空闲核也必须参与
        tla.cross_core_wait_flag(up_ready, tla.arch.MTE2)
        tla.cross_core_set_flag(all_aic, tla.arch.MTE2)
        tla.cross_core_wait_flag(all_aic, tla.arch.MTE2)
        tla.cross_core_set_flag(down_release, tla.arch.MTE2)
        tla.set_flag(down_l1a0_available)
        tla.set_flag(down_l1a1_available)
        tla.set_flag(down_l1b0_available)
        tla.set_flag(down_l1b1_available)
        tla.set_flag(down_l0a0_available)
        tla.set_flag(down_l0a1_available)
        tla.set_flag(down_l0b0_available)
        tla.set_flag(down_l0b1_available)
        tla.set_flag(down_l0c_available)
        down_runtime_zero = tla.as_numeric(0)
        down_l1_buf_idx = down_runtime_zero
        down_l0_buf_idx = down_runtime_zero
        down_block_range = tla.range(tla.arch.block_idx(), down_total_blocks, tla.arch.block_num())
        for down_block_linear in down_block_range:
            down_block_row = down_block_linear // down_grid_n
            down_block_col = down_block_linear % down_grid_n
            down_gm_a_by_core = tla.tile_view(
                hidden,
                tla.make_shape(down_tiling.l1_tm, down_k),
                tla.make_coord(down_block_row, down_c0),
            )
            down_gm_b_by_core = tla.tile_view(
                weight2,
                tla.make_shape(down_k, down_tiling.l1_tn),
                tla.make_coord(down_c0, down_block_col),
            )
            down_gm_c_by_core = tla.tile_view(
                output,
                tla.make_shape(down_tiling.l1_tm, down_tiling.l1_tn),
                tla.make_coord(down_block_row, down_block_col),
            )
            down_k_block = down_gm_a_by_core.origin_shape[1]
            down_k_l1_count = (down_k_block + down_tiling.l1_tk - 1) // down_tiling.l1_tk
            down_k_l1_range = tla.range(down_c0, down_k_l1_count, down_c1)
            down_l0_c = tla.make_tensor_like(down_l0c_ptr, down_gm_c_by_core)
            for down_k_l1 in down_k_l1_range:
                down_gm_a_by_l1 = tla.tile_view(
                    down_gm_a_by_core,
                    tla.make_shape(down_tiling.l1_tm, down_tiling.l1_tk),
                    tla.make_coord(down_c0, down_k_l1),
                )
                down_gm_b_by_l1 = tla.tile_view(
                    down_gm_b_by_core,
                    tla.make_shape(down_tiling.l1_tk, down_tiling.l1_tn),
                    tla.make_coord(down_k_l1, down_c0),
                )
                down_l1_a = tla.make_tensor_like(
                    down_l1a0_ptr if down_l1_buf_idx == down_c0 else down_l1a1_ptr,
                    down_gm_a_by_l1,
                )
                down_l1_b = tla.make_tensor_like(
                    down_l1b0_ptr if down_l1_buf_idx == down_c0 else down_l1b1_ptr,
                    down_gm_b_by_l1,
                )
                if down_l1_buf_idx == down_c0:
                    tla.wait_flag(down_l1a0_available)
                else:
                    tla.wait_flag(down_l1a1_available)
                tla.copy(down_l1_a, down_gm_a_by_l1)
                if down_l1_buf_idx == down_c0:
                    tla.set_flag(down_l1a0_data_ready)
                else:
                    tla.set_flag(down_l1a1_data_ready)
                if down_l1_buf_idx == down_c0:
                    tla.wait_flag(down_l1b0_available)
                else:
                    tla.wait_flag(down_l1b1_available)
                tla.copy(down_l1_b, down_gm_b_by_l1)
                if down_l1_buf_idx == down_c0:
                    tla.set_flag(down_l1b0_data_ready)
                else:
                    tla.set_flag(down_l1b1_data_ready)
                down_k_l0_count = (down_l1_a.origin_shape[1] + down_tiling.l0_tk - 1) // down_tiling.l0_tk
                down_k_l0_range = tla.range(down_c0, down_k_l0_count, down_c1)
                for down_k_l0 in down_k_l0_range:
                    down_l1_a_by_l0 = tla.tile_view(
                        down_l1_a,
                        tla.make_shape(down_tiling.l0_tm, down_tiling.l0_tk),
                        tla.make_coord(down_c0, down_k_l0),
                    )
                    down_l1_b_by_l0 = tla.tile_view(
                        down_l1_b,
                        tla.make_shape(down_tiling.l0_tk, down_tiling.l0_tn),
                        tla.make_coord(down_k_l0, down_c0),
                    )
                    down_l0_a = tla.make_tensor_like(
                        down_l0a0_ptr if down_l0_buf_idx == down_c0 else down_l0a1_ptr,
                        down_l1_a_by_l0,
                    )
                    down_l0_b = tla.make_tensor_like(
                        down_l0b0_ptr if down_l0_buf_idx == down_c0 else down_l0b1_ptr,
                        down_l1_b_by_l0,
                    )
                    if down_k_l0 == 0:
                        if down_l1_buf_idx == down_c0:
                            tla.wait_flag(down_l1a0_data_ready)
                        else:
                            tla.wait_flag(down_l1a1_data_ready)
                    if down_l0_buf_idx == down_c0:
                        tla.wait_flag(down_l0a0_available)
                    else:
                        tla.wait_flag(down_l0a1_available)
                    tla.copy(down_l0_a, down_l1_a_by_l0)
                    if down_k_l0 == down_k_l0_count - 1:
                        if down_l1_buf_idx == down_c0:
                            tla.set_flag(down_l1a0_available)
                        else:
                            tla.set_flag(down_l1a1_available)
                    if down_k_l0 == 0:
                        if down_l1_buf_idx == down_c0:
                            tla.wait_flag(down_l1b0_data_ready)
                        else:
                            tla.wait_flag(down_l1b1_data_ready)
                    if down_l0_buf_idx == down_c0:
                        tla.wait_flag(down_l0b0_available)
                    else:
                        tla.wait_flag(down_l0b1_available)
                    tla.copy(down_l0_b, down_l1_b_by_l0)
                    if down_k_l0 == down_k_l0_count - 1:
                        if down_l1_buf_idx == down_c0:
                            tla.set_flag(down_l1b0_available)
                        else:
                            tla.set_flag(down_l1b1_available)
                    tla.set_flag(down_l0_ab_data_ready)
                    tla.wait_flag(down_l0_ab_data_ready)
                    down_unit_flag = 3 if down_k_l1 == down_k_l1_count - 1 and down_k_l0 == down_k_l0_count - 1 else 2
                    down_init_c = True if down_k_l1 == 0 and down_k_l0 == 0 else False
                    if tla.const_expr(hf32_mode != tla.params.HF32Mode.HF32_DISABLE and down_dtype_a == tla.Float32 and (down_dtype_b == tla.Float32)):
                        tla.mmad(
                            down_l0_c,
                            down_l0_a,
                            down_l0_b,
                            init_c=down_init_c,
                            unit_flag=down_unit_flag,
                            hf32_mode=hf32_mode,
                        )
                    else:
                        tla.mmad(
                            down_l0_c,
                            down_l0_a,
                            down_l0_b,
                            init_c=down_init_c,
                            unit_flag=down_unit_flag,
                        )
                    if down_l0_buf_idx == down_c0:
                        tla.set_flag(down_l0a0_available)
                        tla.set_flag(down_l0b0_available)
                    else:
                        tla.set_flag(down_l0a1_available)
                        tla.set_flag(down_l0b1_available)
                    down_l0_buf_idx = down_c1 - down_l0_buf_idx
                down_l1_buf_idx = down_c1 - down_l1_buf_idx
            tla.copy(down_gm_c_by_core, down_l0_c, tla.params.CopyL0C2DstParams(unit_flag=3))
        tla.wait_flag(down_l1a0_available)
        tla.wait_flag(down_l1a1_available)
        tla.wait_flag(down_l1b0_available)
        tla.wait_flag(down_l1b1_available)
        tla.wait_flag(down_l0a0_available)
        tla.wait_flag(down_l0a1_available)
        tla.wait_flag(down_l0b0_available)
        tla.wait_flag(down_l0b1_available)
        tla.wait_flag(down_l0c_available)
    # AIV：GELU和hidden写回，全部AIV参与阶段屏障
    with tla.vector():
        if tla.arch.sub_block_idx() == 0:
            tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
            for vector_tile in tla.range(tla.arch.block_idx(), up_total_blocks, tla.arch.block_num()):
                tla.cross_core_wait_flag(tile_ready, tla.arch.VECTOR, aiv_id=0)
                with tla.vec.func(mode='simd'):
                    mask32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    mask16 = tla.create_mask(pattern=tla.mask.ALL, dtype=hidden.ptr.dtype)
                    for chunk in tla.range(0, 64, 1):
                        src = tla.tile_view(acc_1d, tla.make_shape(64), tla.make_coord(chunk))
                        dst = tla.tile_view(out_1d, tla.make_shape(64), tla.make_coord(chunk))
                        z = src.load()
                        t = tla.mul(z, 0.7071067811865476, mask=mask32)
                        t = tla.max(t, -3.92, mask=mask32)
                        t = tla.min(t, 3.92, mask=mask32)
                        t2 = tla.mul(t, t, mask=mask32)
                        p = tla.add(tla.mul(t2, 0.053443748819, mask=mask32), 7.5517016694, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 101.62808918, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 1393.8061484, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 5063.791506, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 29639.384698, mask=mask32)
                        p = tla.mul(p, t, mask=mask32)
                        q = tla.add(t2, 31.212858877, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 398.56963806, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 3023.124815, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 13243.365831, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 26267.224157, mask=mask32)
                        y = tla.mul(
                            tla.mul(z, 0.5, mask=mask32),
                            tla.add(tla.div(p, q, mask=mask32), 1.0, mask=mask32),
                            mask=mask32,
                        )
                        dst.store(y.to(hidden.ptr.dtype, cast_params, mask32), store_params, mask=mask16)
                tla.set_flag(gelu_done)
                tla.wait_flag(gelu_done)
                hidden_tile = tla.tile_view(
                    hidden,
                    tla.make_shape(64, 64),
                    tla.make_coord(vector_tile // up_grid_n, vector_tile % up_grid_n),
                )
                tla.copy(hidden_tile, out_ub)
                tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(up_ready, tla.arch.MTE3)
        tla.cross_core_wait_flag(down_release, tla.arch.MTE2)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def ffn_basic_gelu_streamk_kernel(
    x: tla.Tensor,
    weight1: tla.Tensor,
    weight2: tla.Tensor,
    hidden: tla.Tensor,
    output: tla.Tensor,
    workspace: tla.Tensor,
    up_tiling: TilingParams,
    down_tiling: TilingParams,
    swizzle: SwizzleParams,
    block_dim: tla.Constexpr[int],
    hf32_mode: tla.Constexpr[tla.params.HF32Mode],
):
    """Basic上投影→GELU→Stream-K下投影，单Kernel完成

    输入：同设备FP16/BF16的补零X/W1/W2，hidden/output为同dtype缓冲，workspace为FP32
    逻辑形状为X[mp,kp]、W1[kp,hp]、W2[hp,np]，权重底层采用Linear转置存储
    mp/hp/np按64补齐，kp按16补齐，Host检查分块、核数和路径适用性
    输出：写hidden[mp,hp]和output[mp,np]，无Python返回值，Host负责裁剪恢复形状
    """
    up_c0 = 0
    up_c1 = 1
    up_dtype_a = x.ptr.dtype
    up_dtype_b = weight1.ptr.dtype
    up_m = x.origin_shape[0]
    up_n = weight1.origin_shape[1]
    up_k = x.origin_shape[1]
    up_l1a0_data_ready = tla.flag('up_l1a0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1a1_data_ready = tla.flag('up_l1a1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1b0_data_ready = tla.flag('up_l1b0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1b1_data_ready = tla.flag('up_l1b1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    up_l1a0_available = tla.flag('up_l1a0_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1a1_available = tla.flag('up_l1a1_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1b0_available = tla.flag('up_l1b0_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l1b1_available = tla.flag('up_l1b1_available', tla.arch.MTE1, tla.arch.MTE2)
    up_l0a0_available = tla.flag('up_l0a0_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0a1_available = tla.flag('up_l0a1_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0b0_available = tla.flag('up_l0b0_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0b1_available = tla.flag('up_l0b1_available', tla.arch.CUBE, tla.arch.MTE1)
    up_l0_ab_data_ready = tla.flag('up_l0_ab_data_ready', tla.arch.MTE1, tla.arch.CUBE)
    up_l0c_available = tla.flag('up_l0c_available', tla.arch.FIX, tla.arch.CUBE)
    up_l1a0_ptr = tla.allocate(up_tiling.l1_tm * up_tiling.l1_tk, up_dtype_a, tla.AddressSpace.l1, 512)
    up_l1a1_ptr = tla.allocate(up_tiling.l1_tm * up_tiling.l1_tk, up_dtype_a, tla.AddressSpace.l1, 512)
    up_l1b0_ptr = tla.allocate(up_tiling.l1_tk * up_tiling.l1_tn, up_dtype_b, tla.AddressSpace.l1, 512)
    up_l1b1_ptr = tla.allocate(up_tiling.l1_tk * up_tiling.l1_tn, up_dtype_b, tla.AddressSpace.l1, 512)
    up_l0a0_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tk, up_dtype_a, tla.AddressSpace.l0a, 512)
    up_l0a1_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tk, up_dtype_a, tla.AddressSpace.l0a, 512)
    up_l0b0_ptr = tla.allocate(up_tiling.l0_tk * up_tiling.l0_tn, up_dtype_b, tla.AddressSpace.l0b, 512)
    up_l0b1_ptr = tla.allocate(up_tiling.l0_tk * up_tiling.l0_tn, up_dtype_b, tla.AddressSpace.l0b, 512)
    up_l0c_ptr = tla.allocate(up_tiling.l0_tm * up_tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 512)
    up_grid_m = (up_m + up_tiling.l1_tm - 1) // up_tiling.l1_tm
    up_grid_n = (up_n + up_tiling.l1_tn - 1) // up_tiling.l1_tn
    up_total_blocks = up_grid_m * up_grid_n
    down_c0 = 0
    down_c1 = 1
    down_dtype_a = hidden.ptr.dtype
    down_dtype_b = weight2.ptr.dtype
    down_dtype_gm_c = output.ptr.dtype
    down_DTYPE_C = tla.Float32
    down_aiv_m_chunks = AIV_TILE_M // AIV_REG_M
    down_aiv_n_chunks = down_tiling.l1_tn // AIV_REG_N
    down_m = hidden.origin_shape[0]
    down_n = weight2.origin_shape[1]
    down_k = hidden.origin_shape[1]
    down_loops_m = (down_m + down_tiling.l1_tm - 1) // down_tiling.l1_tm
    down_loops_n = (down_n + down_tiling.l1_tn - 1) // down_tiling.l1_tn
    down_loops_k = (down_k + down_tiling.l1_tk - 1) // down_tiling.l1_tk
    down_total_mn = down_loops_m * down_loops_n
    down_streamk_blocks = down_total_mn % block_dim
    down_normal_blocks = down_total_mn - down_streamk_blocks
    down_k_tile_num_per_core = down_streamk_blocks * down_loops_k // block_dim
    down_k_tile_remain = down_streamk_blocks * down_loops_k % block_dim
    down_core_loops = down_total_mn // block_dim * block_dim + min(
        down_streamk_blocks * down_loops_k,
        block_dim,
    )
    down_streamk_cores = down_core_loops - down_normal_blocks
    down_l1a0_data_ready = tla.flag('down_l1a0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1a1_data_ready = tla.flag('down_l1a1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1b0_data_ready = tla.flag('down_l1b0_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1b1_data_ready = tla.flag('down_l1b1_data_ready', tla.arch.MTE2, tla.arch.MTE1)
    down_l1a0_available = tla.flag('down_l1a0_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1a1_available = tla.flag('down_l1a1_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1b0_available = tla.flag('down_l1b0_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l1b1_available = tla.flag('down_l1b1_available', tla.arch.MTE1, tla.arch.MTE2)
    down_l0a0_available = tla.flag('down_l0a0_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0a1_available = tla.flag('down_l0a1_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0b0_available = tla.flag('down_l0b0_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0b1_available = tla.flag('down_l0b1_available', tla.arch.CUBE, tla.arch.MTE1)
    down_l0_ab_data_ready = tla.flag('down_l0_ab_data_ready', tla.arch.MTE1, tla.arch.CUBE)
    down_l0c_available = tla.flag('down_l0c_available', tla.arch.FIX, tla.arch.CUBE)
    down_aic_finish = tla.cross_flag('down_aic_finish', mode=2)
    down_aiv_ibarrier = tla.cross_flag('down_aiv_ibarrier', mode=0)
    down_l1a0_ptr = tla.allocate(
        down_tiling.l1_tm * down_tiling.l1_tk,
        down_dtype_a,
        tla.AddressSpace.l1,
        512,
    )
    down_l1a1_ptr = tla.allocate(
        down_tiling.l1_tm * down_tiling.l1_tk,
        down_dtype_a,
        tla.AddressSpace.l1,
        512,
    )
    down_l1b0_ptr = tla.allocate(
        down_tiling.l1_tk * down_tiling.l1_tn,
        down_dtype_b,
        tla.AddressSpace.l1,
        512,
    )
    down_l1b1_ptr = tla.allocate(
        down_tiling.l1_tk * down_tiling.l1_tn,
        down_dtype_b,
        tla.AddressSpace.l1,
        512,
    )
    down_l0a0_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tk,
        down_dtype_a,
        tla.AddressSpace.l0a,
        512,
    )
    down_l0a1_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tk,
        down_dtype_a,
        tla.AddressSpace.l0a,
        512,
    )
    down_l0b0_ptr = tla.allocate(
        down_tiling.l0_tk * down_tiling.l0_tn,
        down_dtype_b,
        tla.AddressSpace.l0b,
        512,
    )
    down_l0b1_ptr = tla.allocate(
        down_tiling.l0_tk * down_tiling.l0_tn,
        down_dtype_b,
        tla.AddressSpace.l0b,
        512,
    )
    down_l0c_ptr = tla.allocate(
        down_tiling.l0_tm * down_tiling.l0_tn,
        down_DTYPE_C,
        tla.AddressSpace.l0c,
        512,
    )
    down_aiv_ub_tile_elems = AIV_TILE_M * down_tiling.l1_tn
    down_aiv_acc_ptr = tla.allocate(down_aiv_ub_tile_elems, down_DTYPE_C, tla.AddressSpace.ub, 256)
    down_aiv_temp_ptr = tla.allocate(down_aiv_ub_tile_elems, down_DTYPE_C, tla.AddressSpace.ub, 256)
    down_aiv_out_ptr = tla.allocate(down_aiv_ub_tile_elems, down_dtype_gm_c, tla.AddressSpace.ub, 256)
    down_aiv_ub_layout = tla.make_layout(
        tla.make_shape(AIV_TILE_M, down_tiling.l1_tn),
        tla.make_stride(down_tiling.l1_tn, down_c1),
        layoutTag=tla.arch.RowMajor,
    )
    down_aiv_acc_ub = tla.make_tensor(down_aiv_acc_ptr, down_aiv_ub_layout)
    down_aiv_temp_ub = tla.make_tensor(down_aiv_temp_ptr, down_aiv_ub_layout)
    down_aiv_out_ub = tla.make_tensor(down_aiv_out_ptr, down_aiv_ub_layout)
    down_aiv_loaded = tla.flag('down_aiv_loaded', tla.arch.MTE2, tla.arch.VECTOR)
    down_aiv_vec_to_mte2 = tla.flag('down_aiv_vec_to_mte2', tla.arch.VECTOR, tla.arch.MTE2)
    down_aiv_done = tla.flag('down_aiv_done', tla.arch.VECTOR, tla.arch.MTE3)
    tile_available = tla.cross_flag('ffn_tile_available', mode=4)
    tile_ready = tla.cross_flag('ffn_tile_ready', mode=4)
    up_ready = tla.cross_flag('ffn_up_ready', mode=2)
    all_aic = tla.cross_flag('ffn_all_aic', mode=0)
    down_release = tla.cross_flag('ffn_down_release', mode=2)
    gelu_done = tla.flag('ffn_gelu_done', tla.arch.VECTOR, tla.arch.MTE3)
    acc_ptr = tla.allocate(4096, tla.Float32, tla.AddressSpace.ub, 256)
    out_ptr = tla.allocate(4096, hidden.ptr.dtype, tla.AddressSpace.ub, 256)
    ub_layout = tla.make_layout(tla.make_shape(64, 64), tla.make_stride(64, 1), layoutTag=tla.arch.RowMajor)
    acc_ub = tla.make_tensor(acc_ptr, ub_layout)
    out_ub = tla.make_tensor(out_ptr, ub_layout)
    acc_1d = tla.make_tensor(acc_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
    out_1d = tla.make_tensor(out_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
    cast_params = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    store_params = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
    # AIC：上投影、L0C→UB、阶段屏障、下投影
    with tla.cube():
        tla.set_flag(up_l1a0_available)
        tla.set_flag(up_l1a1_available)
        tla.set_flag(up_l1b0_available)
        tla.set_flag(up_l1b1_available)
        tla.set_flag(up_l0a0_available)
        tla.set_flag(up_l0a1_available)
        tla.set_flag(up_l0b0_available)
        tla.set_flag(up_l0b1_available)
        tla.set_flag(up_l0c_available)
        up_runtime_zero = tla.as_numeric(0)
        up_l1_buf_idx = up_runtime_zero
        up_l0_buf_idx = up_runtime_zero
        up_block_range = tla.range(tla.arch.block_idx(), up_total_blocks, tla.arch.block_num())
        for up_block_linear in up_block_range:
            up_block_row = up_block_linear // up_grid_n
            up_block_col = up_block_linear % up_grid_n
            up_gm_a_by_core = tla.tile_view(
                x,
                tla.make_shape(up_tiling.l1_tm, up_k),
                tla.make_coord(up_block_row, up_c0),
            )
            up_gm_b_by_core = tla.tile_view(
                weight1,
                tla.make_shape(up_k, up_tiling.l1_tn),
                tla.make_coord(up_c0, up_block_col),
            )
            up_gm_c_by_core = tla.tile_view(
                hidden,
                tla.make_shape(up_tiling.l1_tm, up_tiling.l1_tn),
                tla.make_coord(up_block_row, up_block_col),
            )
            up_k_block = up_gm_a_by_core.origin_shape[1]
            up_k_l1_count = (up_k_block + up_tiling.l1_tk - 1) // up_tiling.l1_tk
            up_k_l1_range = tla.range(up_c0, up_k_l1_count, up_c1)
            up_l0_c = tla.make_tensor_like(up_l0c_ptr, up_gm_c_by_core)
            for up_k_l1 in up_k_l1_range:
                up_gm_a_by_l1 = tla.tile_view(
                    up_gm_a_by_core,
                    tla.make_shape(up_tiling.l1_tm, up_tiling.l1_tk),
                    tla.make_coord(up_c0, up_k_l1),
                )
                up_gm_b_by_l1 = tla.tile_view(
                    up_gm_b_by_core,
                    tla.make_shape(up_tiling.l1_tk, up_tiling.l1_tn),
                    tla.make_coord(up_k_l1, up_c0),
                )
                up_l1_a = tla.make_tensor_like(
                    up_l1a0_ptr if up_l1_buf_idx == up_c0 else up_l1a1_ptr,
                    up_gm_a_by_l1,
                )
                up_l1_b = tla.make_tensor_like(
                    up_l1b0_ptr if up_l1_buf_idx == up_c0 else up_l1b1_ptr,
                    up_gm_b_by_l1,
                )
                if up_l1_buf_idx == up_c0:
                    tla.wait_flag(up_l1a0_available)
                else:
                    tla.wait_flag(up_l1a1_available)
                tla.copy(up_l1_a, up_gm_a_by_l1)
                if up_l1_buf_idx == up_c0:
                    tla.set_flag(up_l1a0_data_ready)
                else:
                    tla.set_flag(up_l1a1_data_ready)
                if up_l1_buf_idx == up_c0:
                    tla.wait_flag(up_l1b0_available)
                else:
                    tla.wait_flag(up_l1b1_available)
                tla.copy(up_l1_b, up_gm_b_by_l1)
                if up_l1_buf_idx == up_c0:
                    tla.set_flag(up_l1b0_data_ready)
                else:
                    tla.set_flag(up_l1b1_data_ready)
                up_k_l0_count = (up_l1_a.origin_shape[1] + up_tiling.l0_tk - 1) // up_tiling.l0_tk
                up_k_l0_range = tla.range(up_c0, up_k_l0_count, up_c1)
                for up_k_l0 in up_k_l0_range:
                    up_l1_a_by_l0 = tla.tile_view(
                        up_l1_a,
                        tla.make_shape(up_tiling.l0_tm, up_tiling.l0_tk),
                        tla.make_coord(up_c0, up_k_l0),
                    )
                    up_l1_b_by_l0 = tla.tile_view(
                        up_l1_b,
                        tla.make_shape(up_tiling.l0_tk, up_tiling.l0_tn),
                        tla.make_coord(up_k_l0, up_c0),
                    )
                    up_l0_a = tla.make_tensor_like(
                        up_l0a0_ptr if up_l0_buf_idx == up_c0 else up_l0a1_ptr,
                        up_l1_a_by_l0,
                    )
                    up_l0_b = tla.make_tensor_like(
                        up_l0b0_ptr if up_l0_buf_idx == up_c0 else up_l0b1_ptr,
                        up_l1_b_by_l0,
                    )
                    if up_k_l0 == 0:
                        if up_l1_buf_idx == up_c0:
                            tla.wait_flag(up_l1a0_data_ready)
                        else:
                            tla.wait_flag(up_l1a1_data_ready)
                    if up_l0_buf_idx == up_c0:
                        tla.wait_flag(up_l0a0_available)
                    else:
                        tla.wait_flag(up_l0a1_available)
                    tla.copy(up_l0_a, up_l1_a_by_l0)
                    if up_k_l0 == up_k_l0_count - 1:
                        if up_l1_buf_idx == up_c0:
                            tla.set_flag(up_l1a0_available)
                        else:
                            tla.set_flag(up_l1a1_available)
                    if up_k_l0 == 0:
                        if up_l1_buf_idx == up_c0:
                            tla.wait_flag(up_l1b0_data_ready)
                        else:
                            tla.wait_flag(up_l1b1_data_ready)
                    if up_l0_buf_idx == up_c0:
                        tla.wait_flag(up_l0b0_available)
                    else:
                        tla.wait_flag(up_l0b1_available)
                    tla.copy(up_l0_b, up_l1_b_by_l0)
                    if up_k_l0 == up_k_l0_count - 1:
                        if up_l1_buf_idx == up_c0:
                            tla.set_flag(up_l1b0_available)
                        else:
                            tla.set_flag(up_l1b1_available)
                    tla.set_flag(up_l0_ab_data_ready)
                    tla.wait_flag(up_l0_ab_data_ready)
                    up_unit_flag = 3 if up_k_l1 == up_k_l1_count - 1 and up_k_l0 == up_k_l0_count - 1 else 2
                    up_init_c = True if up_k_l1 == 0 and up_k_l0 == 0 else False
                    if tla.const_expr(hf32_mode != tla.params.HF32Mode.HF32_DISABLE and up_dtype_a == tla.Float32 and (up_dtype_b == tla.Float32)):
                        tla.mmad(
                            up_l0_c,
                            up_l0_a,
                            up_l0_b,
                            init_c=up_init_c,
                            unit_flag=up_unit_flag,
                            hf32_mode=hf32_mode,
                        )
                    else:
                        tla.mmad(up_l0_c, up_l0_a, up_l0_b, init_c=up_init_c, unit_flag=up_unit_flag)
                    if up_l0_buf_idx == up_c0:
                        tla.set_flag(up_l0a0_available)
                        tla.set_flag(up_l0b0_available)
                    else:
                        tla.set_flag(up_l0a1_available)
                        tla.set_flag(up_l0b1_available)
                    up_l0_buf_idx = up_c1 - up_l0_buf_idx
                up_l1_buf_idx = up_c1 - up_l1_buf_idx
            tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
            tla.copy(
                acc_ub,
                up_l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=3, l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0),
            )
            tla.cross_core_set_flag(tile_ready, tla.arch.FIX, aiv_id=0)
        tla.wait_flag(up_l1a0_available)
        tla.wait_flag(up_l1a1_available)
        tla.wait_flag(up_l1b0_available)
        tla.wait_flag(up_l1b1_available)
        tla.wait_flag(up_l0a0_available)
        tla.wait_flag(up_l0a1_available)
        tla.wait_flag(up_l0b0_available)
        tla.wait_flag(up_l0b1_available)
        tla.wait_flag(up_l0c_available)
        tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)
        # mode2报到→mode0全AIC汇合→mode2放行，空闲核也必须参与
        tla.cross_core_wait_flag(up_ready, tla.arch.MTE2)
        tla.cross_core_set_flag(all_aic, tla.arch.MTE2)
        tla.cross_core_wait_flag(all_aic, tla.arch.MTE2)
        tla.cross_core_set_flag(down_release, tla.arch.MTE2)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.set_flag(down_l1a0_available)
        tla.set_flag(down_l1a1_available)
        tla.set_flag(down_l1b0_available)
        tla.set_flag(down_l1b1_available)
        tla.set_flag(down_l0a0_available)
        tla.set_flag(down_l0a1_available)
        tla.set_flag(down_l0b0_available)
        tla.set_flag(down_l0b1_available)
        tla.set_flag(down_l0c_available)
        down_l1_buf_idx = down_c0
        down_l0_buf_idx = down_c0
        down_block_idx = tla.arch.block_idx()
        if down_block_idx >= down_streamk_cores:
            tla.cross_core_set_flag(down_aic_finish, tla.arch.FIX)
        down_loop_range = tla.range(down_block_idx, down_core_loops, tla.arch.block_num())
        for down_loop_idx in down_loop_range:
            down_actual_loop_idx = down_loop_idx
            if down_normal_blocks > 0:
                if down_block_idx < down_streamk_cores:
                    down_swap_at = down_normal_blocks - tla.arch.block_num() + down_block_idx
                    if down_loop_idx == down_swap_at:
                        down_actual_loop_idx = down_normal_blocks + down_block_idx
                    elif down_loop_idx >= down_normal_blocks:
                        down_actual_loop_idx = down_swap_at
                elif down_loop_idx >= down_normal_blocks:
                    down_actual_loop_idx = down_normal_blocks - tla.arch.block_num() + down_block_idx
            down_is_sk_now = False
            if down_normal_blocks > 0:
                if down_actual_loop_idx >= down_normal_blocks:
                    down_is_sk_now = True
            else:
                down_is_sk_now = True
            down_swizzle_span = swizzle.SWIZZLE_OFFSET * down_loops_n
            down_swizzle_tb_loop = (down_loops_m + swizzle.SWIZZLE_OFFSET - 1) // swizzle.SWIZZLE_OFFSET
            down_tile_block_idx = down_actual_loop_idx // down_swizzle_span
            down_in_tile = down_actual_loop_idx % down_swizzle_span
            down_n_row = swizzle.SWIZZLE_OFFSET
            if down_tile_block_idx == down_swizzle_tb_loop - 1:
                down_n_row = down_loops_m - swizzle.SWIZZLE_OFFSET * down_tile_block_idx
            down_block_row = down_tile_block_idx * swizzle.SWIZZLE_OFFSET + down_in_tile % down_n_row
            down_block_col = down_in_tile // down_n_row
            if down_tile_block_idx % 2 == 1:
                down_block_col = down_loops_n - down_block_col - 1
            down_block_k = down_c0
            down_block_actual_k = down_k
            down_streamk_block_row = down_block_row
            down_streamk_block_col = down_block_col
            down_streamk_block_k = down_c0
            down_streamk_actual_k = down_c0
            down_sk_task_id = down_block_idx
            down_sk_slot_count = 1
            if down_is_sk_now:
                down_rel = down_actual_loop_idx - down_normal_blocks
                down_cur_k_tile_num = down_k_tile_num_per_core
                down_k_tile_idx = down_rel * down_k_tile_num_per_core + down_k_tile_remain
                if down_rel < down_k_tile_remain:
                    down_cur_k_tile_num = down_k_tile_num_per_core + 1
                    down_k_tile_idx = down_rel * down_cur_k_tile_num
                down_streamk_block_idx = down_k_tile_idx // down_loops_k
                down_block_linear = down_normal_blocks + down_streamk_block_idx
                down_block_tb_idx = down_block_linear // down_swizzle_span
                down_block_in_tile = down_block_linear % down_swizzle_span
                down_block_n_row = swizzle.SWIZZLE_OFFSET
                if down_block_tb_idx == down_swizzle_tb_loop - 1:
                    down_block_n_row = down_loops_m - swizzle.SWIZZLE_OFFSET * down_block_tb_idx
                down_block_row = down_block_tb_idx * swizzle.SWIZZLE_OFFSET + down_block_in_tile % down_block_n_row
                down_block_col = down_block_in_tile // down_block_n_row
                if down_block_tb_idx % 2 == 1:
                    down_block_col = down_loops_n - down_block_col - 1
                down_block_k = down_k_tile_idx % down_loops_k
                down_block_actual_k = down_cur_k_tile_num * down_tiling.l1_tk
                if (down_k_tile_idx % down_loops_k + down_cur_k_tile_num) * down_tiling.l1_tk > down_k:
                    down_block_actual_k = down_k - down_k_tile_idx % down_loops_k * down_tiling.l1_tk
                down_streamk_block_row = down_block_row
                down_streamk_block_col = down_block_col
                down_streamk_block_k = down_c0
                down_streamk_actual_k = down_c0
                if down_k_tile_idx % down_loops_k + down_cur_k_tile_num > down_loops_k:
                    down_sk_slot_count = 2
                    down_next_streamk_block_idx = (down_k_tile_idx + down_cur_k_tile_num) // down_loops_k
                    down_streamk_linear = down_normal_blocks + down_next_streamk_block_idx
                    down_streamk_tb_idx = down_streamk_linear // down_swizzle_span
                    down_streamk_in_tile = down_streamk_linear % down_swizzle_span
                    down_streamk_n_row = swizzle.SWIZZLE_OFFSET
                    if down_streamk_tb_idx == down_swizzle_tb_loop - 1:
                        down_streamk_n_row = down_loops_m - swizzle.SWIZZLE_OFFSET * down_streamk_tb_idx
                    down_streamk_block_row = down_streamk_tb_idx * swizzle.SWIZZLE_OFFSET + down_streamk_in_tile % down_streamk_n_row
                    down_streamk_block_col = down_streamk_in_tile // down_streamk_n_row
                    if down_streamk_tb_idx % 2 == 1:
                        down_streamk_block_col = down_loops_n - down_streamk_block_col - 1
                    down_streamk_block_k = down_c0
                    down_streamk_actual_k = (down_k_tile_idx + down_cur_k_tile_num) % down_loops_k * down_tiling.l1_tk
            down_sk_slots = tla.range(down_c0, down_sk_slot_count, down_c1)
            for down_sk_slot in down_sk_slots:
                down_sk_slot_row = down_block_row if down_sk_slot == 0 else down_streamk_block_row
                down_sk_slot_col = down_block_col if down_sk_slot == 0 else down_streamk_block_col
                down_sk_slot_block_k = down_block_k if down_sk_slot == 0 else down_streamk_block_k
                down_sk_slot_actual_k = down_block_actual_k if down_sk_slot == 0 else down_streamk_actual_k
                down_sk_slot_ws_row = down_sk_task_id * 2 + down_sk_slot
                down_gm_a_by_core = tla.tile_view(
                    hidden,
                    tla.make_shape(down_tiling.l1_tm, down_k),
                    tla.make_coord(down_sk_slot_row, down_c0),
                )
                down_gm_b_by_core = tla.tile_view(
                    weight2,
                    tla.make_shape(down_k, down_tiling.l1_tn),
                    tla.make_coord(down_c0, down_sk_slot_col),
                )
                down_gm_c_by_core = tla.tile_view(
                    output,
                    tla.make_shape(down_tiling.l1_tm, down_tiling.l1_tn),
                    tla.make_coord(down_sk_slot_row, down_sk_slot_col),
                )
                down_gm_ws_by_core = tla.tile_view(
                    workspace,
                    tla.make_shape(down_tiling.l1_tm, down_tiling.l1_tn),
                    tla.make_coord(down_sk_slot_ws_row, down_c0),
                )
                down_l0_c = tla.make_tensor_like(down_l0c_ptr, down_gm_c_by_core)
                down_k_l1_count = (down_sk_slot_actual_k + down_tiling.l1_tk - 1) // down_tiling.l1_tk
                down_k_l1_range = tla.range(down_c0, down_k_l1_count, down_c1)
                for down_k_l1_i in down_k_l1_range:
                    down_k_l1 = down_sk_slot_block_k + down_k_l1_i
                    down_gm_a_l1 = tla.tile_view(
                        down_gm_a_by_core,
                        tla.make_shape(down_tiling.l1_tm, down_tiling.l1_tk),
                        tla.make_coord(down_c0, down_k_l1),
                    )
                    down_gm_b_l1 = tla.tile_view(
                        down_gm_b_by_core,
                        tla.make_shape(down_tiling.l1_tk, down_tiling.l1_tn),
                        tla.make_coord(down_k_l1, down_c0),
                    )
                    down_l1_a = tla.make_tensor_like(
                        down_l1a0_ptr if down_l1_buf_idx == down_c0 else down_l1a1_ptr,
                        down_gm_a_l1,
                    )
                    down_l1_b = tla.make_tensor_like(
                        down_l1b0_ptr if down_l1_buf_idx == down_c0 else down_l1b1_ptr,
                        down_gm_b_l1,
                    )
                    if down_l1_buf_idx == down_c0:
                        tla.wait_flag(down_l1a0_available)
                    else:
                        tla.wait_flag(down_l1a1_available)
                    tla.copy(down_l1_a, down_gm_a_l1)
                    if down_l1_buf_idx == down_c0:
                        tla.set_flag(down_l1a0_data_ready)
                    else:
                        tla.set_flag(down_l1a1_data_ready)
                    if down_l1_buf_idx == down_c0:
                        tla.wait_flag(down_l1b0_available)
                    else:
                        tla.wait_flag(down_l1b1_available)
                    tla.copy(down_l1_b, down_gm_b_l1)
                    if down_l1_buf_idx == down_c0:
                        tla.set_flag(down_l1b0_data_ready)
                    else:
                        tla.set_flag(down_l1b1_data_ready)
                    down_k_l0_count = (down_l1_a.origin_shape[1] + down_tiling.l0_tk - 1) // down_tiling.l0_tk
                    down_k_l0_range = tla.range(down_c0, down_k_l0_count, down_c1)
                    for down_k_l0 in down_k_l0_range:
                        down_l1_a_l0 = tla.tile_view(
                            down_l1_a,
                            tla.make_shape(down_tiling.l0_tm, down_tiling.l0_tk),
                            tla.make_coord(down_c0, down_k_l0),
                        )
                        down_l1_b_l0 = tla.tile_view(
                            down_l1_b,
                            tla.make_shape(down_tiling.l0_tk, down_tiling.l0_tn),
                            tla.make_coord(down_k_l0, down_c0),
                        )
                        down_l0_a = tla.make_tensor_like(
                            down_l0a0_ptr if down_l0_buf_idx == down_c0 else down_l0a1_ptr,
                            down_l1_a_l0,
                        )
                        down_l0_b = tla.make_tensor_like(
                            down_l0b0_ptr if down_l0_buf_idx == down_c0 else down_l0b1_ptr,
                            down_l1_b_l0,
                        )
                        if down_k_l0 == 0:
                            if down_l1_buf_idx == down_c0:
                                tla.wait_flag(down_l1a0_data_ready)
                            else:
                                tla.wait_flag(down_l1a1_data_ready)
                        if down_l0_buf_idx == down_c0:
                            tla.wait_flag(down_l0a0_available)
                        else:
                            tla.wait_flag(down_l0a1_available)
                        tla.copy(down_l0_a, down_l1_a_l0)
                        if down_k_l0 == down_k_l0_count - 1:
                            if down_l1_buf_idx == down_c0:
                                tla.set_flag(down_l1a0_available)
                            else:
                                tla.set_flag(down_l1a1_available)
                        if down_k_l0 == 0:
                            if down_l1_buf_idx == down_c0:
                                tla.wait_flag(down_l1b0_data_ready)
                            else:
                                tla.wait_flag(down_l1b1_data_ready)
                        if down_l0_buf_idx == down_c0:
                            tla.wait_flag(down_l0b0_available)
                        else:
                            tla.wait_flag(down_l0b1_available)
                        tla.copy(down_l0_b, down_l1_b_l0)
                        if down_k_l0 == down_k_l0_count - 1:
                            if down_l1_buf_idx == down_c0:
                                tla.set_flag(down_l1b0_available)
                            else:
                                tla.set_flag(down_l1b1_available)
                        tla.set_flag(down_l0_ab_data_ready)
                        tla.wait_flag(down_l0_ab_data_ready)
                        down_unit_flag = 3 if down_k_l1_i == down_k_l1_count - 1 and down_k_l0 == down_k_l0_count - 1 else 2
                        down_init_c = True if down_k_l1_i == 0 and down_k_l0 == 0 else False
                        tla.mmad(
                            down_l0_c,
                            down_l0_a,
                            down_l0_b,
                            init_c=down_init_c,
                            unit_flag=down_unit_flag,
                        )
                        if down_l0_buf_idx == down_c0:
                            tla.set_flag(down_l0a0_available)
                            tla.set_flag(down_l0b0_available)
                        else:
                            tla.set_flag(down_l0a1_available)
                            tla.set_flag(down_l0b1_available)
                        down_l0_buf_idx = down_c1 - down_l0_buf_idx
                    down_l1_buf_idx = down_c1 - down_l1_buf_idx
                if down_is_sk_now:
                    tla.copy(down_gm_ws_by_core, down_l0_c, tla.params.CopyL0C2DstParams(unit_flag=3))
                else:
                    tla.copy(down_gm_c_by_core, down_l0_c, tla.params.CopyL0C2DstParams(unit_flag=3))
            if down_is_sk_now:
                if down_normal_blocks > 0:
                    if down_block_idx < down_streamk_cores:
                        if down_loop_idx == down_normal_blocks - tla.arch.block_num() + down_block_idx:
                            tla.cross_core_set_flag(down_aic_finish, tla.arch.FIX)
                if down_normal_blocks == 0:
                    tla.cross_core_set_flag(down_aic_finish, tla.arch.FIX)
        tla.wait_flag(down_l1a0_available)
        tla.wait_flag(down_l1a1_available)
        tla.wait_flag(down_l1b0_available)
        tla.wait_flag(down_l1b1_available)
        tla.wait_flag(down_l0a0_available)
        tla.wait_flag(down_l0a1_available)
        tla.wait_flag(down_l0b0_available)
        tla.wait_flag(down_l0b1_available)
        tla.wait_flag(down_l0c_available)
        tla.pipe_barrier(tla.pipes.ALL)
    # AIV：GELU和hidden写回，全部AIV参与阶段屏障
    with tla.vector():
        if tla.arch.sub_block_idx() == 0:
            tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
            for vector_tile in tla.range(tla.arch.block_idx(), up_total_blocks, tla.arch.block_num()):
                tla.cross_core_wait_flag(tile_ready, tla.arch.VECTOR, aiv_id=0)
                with tla.vec.func(mode='simd'):
                    mask32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    mask16 = tla.create_mask(pattern=tla.mask.ALL, dtype=hidden.ptr.dtype)
                    for chunk in tla.range(0, 64, 1):
                        src = tla.tile_view(acc_1d, tla.make_shape(64), tla.make_coord(chunk))
                        dst = tla.tile_view(out_1d, tla.make_shape(64), tla.make_coord(chunk))
                        z = src.load()
                        t = tla.mul(z, 0.7071067811865476, mask=mask32)
                        t = tla.max(t, -3.92, mask=mask32)
                        t = tla.min(t, 3.92, mask=mask32)
                        t2 = tla.mul(t, t, mask=mask32)
                        p = tla.add(tla.mul(t2, 0.053443748819, mask=mask32), 7.5517016694, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 101.62808918, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 1393.8061484, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 5063.791506, mask=mask32)
                        p = tla.add(tla.mul(p, t2, mask=mask32), 29639.384698, mask=mask32)
                        p = tla.mul(p, t, mask=mask32)
                        q = tla.add(t2, 31.212858877, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 398.56963806, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 3023.124815, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 13243.365831, mask=mask32)
                        q = tla.add(tla.mul(q, t2, mask=mask32), 26267.224157, mask=mask32)
                        y = tla.mul(
                            tla.mul(z, 0.5, mask=mask32),
                            tla.add(tla.div(p, q, mask=mask32), 1.0, mask=mask32),
                            mask=mask32,
                        )
                        dst.store(y.to(hidden.ptr.dtype, cast_params, mask32), store_params, mask=mask16)
                tla.set_flag(gelu_done)
                tla.wait_flag(gelu_done)
                hidden_tile = tla.tile_view(
                    hidden,
                    tla.make_shape(64, 64),
                    tla.make_coord(vector_tile // up_grid_n, vector_tile % up_grid_n),
                )
                tla.copy(hidden_tile, out_ub)
                tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(up_ready, tla.arch.MTE3)
        tla.cross_core_wait_flag(down_release, tla.arch.MTE2)
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_wait_flag(down_aic_finish, tla.arch.MTE2)
        tla.cross_core_set_flag(down_aiv_ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(down_aiv_ibarrier, tla.arch.MTE2)
        down_aiv_id = tla.arch.block_idx()
        down_aiv_sub = tla.arch.sub_block_idx()
        down_aiv_global = down_aiv_id * AIV_SUB_BLOCK_NUM + down_aiv_sub
        for down_aiv_sk_id in tla.range(down_c0, down_streamk_blocks, down_c1):
            down_aiv_start_core = down_c0
            down_aiv_end_core = down_c0
            down_aiv_head_cross = False
            down_aiv_tail_cross = False
            if down_k_tile_num_per_core == 0:
                down_aiv_start_core = down_aiv_sk_id * down_loops_k
                down_aiv_end_core = (down_aiv_sk_id + 1) * down_loops_k
                down_aiv_head_cross = False
                down_aiv_tail_cross = False
            else:
                down_aiv_threshold = down_k_tile_remain * (down_k_tile_num_per_core + 1)
                down_aiv_start_core = down_aiv_sk_id * down_loops_k // (down_k_tile_num_per_core + 1)
                if down_aiv_sk_id * down_loops_k > down_aiv_threshold:
                    down_aiv_start_core = down_k_tile_remain + (down_aiv_sk_id * down_loops_k - down_aiv_threshold) // down_k_tile_num_per_core
                down_aiv_end_core = (down_aiv_sk_id + 1) * down_loops_k // (down_k_tile_num_per_core + 1)
                if (down_aiv_sk_id + 1) * down_loops_k > down_aiv_threshold:
                    down_aiv_end_core = down_k_tile_remain + ((down_aiv_sk_id + 1) * down_loops_k - down_aiv_threshold) // down_k_tile_num_per_core
                down_aiv_head_cross = down_aiv_sk_id * down_loops_k % (down_k_tile_num_per_core + 1) != 0
                if down_aiv_sk_id * down_loops_k > down_aiv_threshold:
                    down_aiv_head_numer = down_aiv_sk_id * down_loops_k - down_aiv_threshold
                    down_aiv_head_cross = down_aiv_head_numer % down_k_tile_num_per_core != 0
                down_aiv_tail_cross = (down_aiv_sk_id + 1) * down_loops_k % (down_k_tile_num_per_core + 1) != 0
                if (down_aiv_sk_id + 1) * down_loops_k > down_aiv_threshold:
                    down_aiv_tail_numer = (down_aiv_sk_id + 1) * down_loops_k - down_aiv_threshold
                    down_aiv_tail_cross = down_aiv_tail_numer % down_k_tile_num_per_core != 0
            down_aiv_end_core_raw = down_aiv_end_core
            down_aiv_labor = (down_aiv_end_core_raw - down_aiv_start_core) * AIV_SUB_BLOCK_NUM
            if down_aiv_tail_cross:
                down_aiv_end_core = down_aiv_end_core + 1
            down_aiv_linear = down_normal_blocks + down_aiv_sk_id
            down_aiv_span = swizzle.SWIZZLE_OFFSET * down_loops_n
            down_aiv_tb_loop = (down_loops_m + swizzle.SWIZZLE_OFFSET - 1) // swizzle.SWIZZLE_OFFSET
            down_aiv_tb_idx = down_aiv_linear // down_aiv_span
            down_aiv_in_tile = down_aiv_linear % down_aiv_span
            down_aiv_n_row = swizzle.SWIZZLE_OFFSET
            if down_aiv_tb_idx == down_aiv_tb_loop - 1:
                down_aiv_n_row = down_loops_m - swizzle.SWIZZLE_OFFSET * down_aiv_tb_idx
            down_aiv_block_row = down_aiv_tb_idx * swizzle.SWIZZLE_OFFSET + down_aiv_in_tile % down_aiv_n_row
            down_aiv_block_col = down_aiv_in_tile // down_aiv_n_row
            if down_aiv_tb_idx % 2 == 1:
                down_aiv_block_col = down_loops_n - down_aiv_block_col - 1
            down_aiv_tile_m = down_tiling.l1_tm
            if down_aiv_block_row == down_loops_m - 1:
                down_aiv_tile_m = down_m - down_aiv_block_row * down_tiling.l1_tm
            down_aiv_slice_count = down_aiv_end_core - down_aiv_start_core
            down_aiv_m_loops = (down_aiv_tile_m + AIV_TILE_M - 1) // AIV_TILE_M
            down_aiv_rows_per_slot = down_tiling.l1_tm // AIV_TILE_M
            if down_aiv_id >= down_aiv_start_core:
                if down_aiv_id < down_aiv_end_core_raw:
                    down_aiv_loop_start = down_aiv_global - down_aiv_start_core * AIV_SUB_BLOCK_NUM
                    down_aiv_chunk_per = (down_aiv_m_loops + down_aiv_labor - 1) // down_aiv_labor
                    down_aiv_chunk_lo = down_aiv_loop_start * down_aiv_chunk_per
                    down_aiv_chunk_hi = down_aiv_chunk_lo + down_aiv_chunk_per
                    if down_aiv_chunk_lo > down_aiv_m_loops:
                        down_aiv_chunk_lo = down_aiv_m_loops
                    if down_aiv_chunk_hi > down_aiv_m_loops:
                        down_aiv_chunk_hi = down_aiv_m_loops
                    for down_aiv_m_idx in tla.range(down_aiv_chunk_lo, down_aiv_chunk_hi, down_c1):
                        down_aiv_c_row = down_aiv_block_row * down_aiv_rows_per_slot + down_aiv_m_idx
                        down_aiv_c_col = down_aiv_block_col
                        down_aiv_gm_c_tile = tla.tile_view(
                            output,
                            tla.make_shape(AIV_TILE_M, down_tiling.l1_tn),
                            tla.make_coord(down_aiv_c_row, down_aiv_c_col),
                        )
                        down_aiv_init_pingpong = 0
                        if down_aiv_head_cross:
                            down_aiv_init_pingpong = 1
                        down_aiv_init_row = (down_aiv_start_core * 2 + down_aiv_init_pingpong) * down_aiv_rows_per_slot + down_aiv_m_idx
                        down_aiv_ws_init = tla.tile_view(
                            workspace,
                            tla.make_shape(AIV_TILE_M, down_tiling.l1_tn),
                            tla.make_coord(down_aiv_init_row, down_c0),
                        )
                        tla.copy(down_aiv_acc_ub, down_aiv_ws_init)
                        tla.set_flag(down_aiv_loaded)
                        tla.wait_flag(down_aiv_loaded)
                        down_aiv_slice_range = tla.range(down_c1, down_aiv_slice_count, down_c1)
                        for down_aiv_slice_idx in down_aiv_slice_range:
                            down_aiv_core = down_aiv_start_core + down_aiv_slice_idx
                            down_aiv_ws_row = down_aiv_core * 2 * down_aiv_rows_per_slot + down_aiv_m_idx
                            down_aiv_ws_tile = tla.tile_view(
                                workspace,
                                tla.make_shape(AIV_TILE_M, down_tiling.l1_tn),
                                tla.make_coord(down_aiv_ws_row, down_c0),
                            )
                            tla.copy(down_aiv_temp_ub, down_aiv_ws_tile)
                            tla.set_flag(down_aiv_loaded)
                            tla.wait_flag(down_aiv_loaded)
                            with tla.vec.func(mode='simd'):
                                for down__aiv_rm in tla.range(0, down_aiv_m_chunks, 1):
                                    for down__aiv_rn in tla.range(0, down_aiv_n_chunks, 1):
                                        down_aiv_acc_chunk = tla.tile_view(
                                            down_aiv_acc_ub,
                                            tla.make_shape(AIV_REG_M, AIV_REG_N),
                                            tla.make_coord(down__aiv_rm, down__aiv_rn),
                                        )
                                        down_aiv_temp_chunk = tla.tile_view(
                                            down_aiv_temp_ub,
                                            tla.make_shape(AIV_REG_M, AIV_REG_N),
                                            tla.make_coord(down__aiv_rm, down__aiv_rn),
                                        )
                                        down_aiv_acc_chunk.store(
                                            tla.add(
                                                down_aiv_acc_chunk.load(),
                                                down_aiv_temp_chunk.load(),
                                                mask=tla.create_mask(
                                                    pattern=tla.mask.ALL,
                                                    dtype=down_DTYPE_C,
                                                ),
                                            ),
                                            mask=tla.create_mask(pattern=tla.mask.ALL, dtype=down_DTYPE_C),
                                        )
                            tla.set_flag(down_aiv_vec_to_mte2)
                            tla.wait_flag(down_aiv_vec_to_mte2)
                        down_aiv_store_ub = down_aiv_acc_ub
                        if tla.const_expr(down_dtype_gm_c != down_DTYPE_C):
                            if tla.const_expr(down_dtype_gm_c == tla.Float16):
                                down_aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_ROUND,
                                )
                            else:
                                down_aiv_cast_even = tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                    round_mode=tla.params.RoundMode.CAST_ROUND,
                                )
                            down_aiv_pack_store = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
                            down_aiv_cast_vl_loops = down_aiv_m_chunks * down_aiv_n_chunks
                            down_aiv_acc_1d = tla.make_tensor(
                                down_aiv_acc_ptr,
                                tla.make_layout(
                                    tla.make_shape(down_aiv_ub_tile_elems),
                                    tla.make_stride(down_c1),
                                ),
                            )
                            down_aiv_out_1d = tla.make_tensor(
                                down_aiv_out_ptr,
                                tla.make_layout(
                                    tla.make_shape(down_aiv_ub_tile_elems),
                                    tla.make_stride(down_c1),
                                ),
                            )
                            with tla.vec.func(mode='simd'):
                                down_aiv_cast_mask = tla.create_mask(
                                    pattern=tla.mask.ALL,
                                    dtype=down_DTYPE_C,
                                )
                                down_aiv_store_mask = tla.create_mask(
                                    pattern=tla.mask.ALL,
                                    dtype=down_dtype_gm_c,
                                )
                                for down_aiv_cast_vl in tla.range(0, down_aiv_cast_vl_loops, 1):
                                    down_aiv_cast_src = tla.tile_view(
                                        down_aiv_acc_1d,
                                        tla.make_shape(AIV_REG_N),
                                        tla.make_coord(down_aiv_cast_vl),
                                    )
                                    down_aiv_cast_dst = tla.tile_view(
                                        down_aiv_out_1d,
                                        tla.make_shape(AIV_REG_N),
                                        tla.make_coord(down_aiv_cast_vl),
                                    )
                                    down_aiv_cast_h = down_aiv_cast_src.load().to(down_dtype_gm_c, down_aiv_cast_even, down_aiv_cast_mask)
                                    down_aiv_cast_dst.store(
                                        down_aiv_cast_h,
                                        down_aiv_pack_store,
                                        mask=down_aiv_store_mask,
                                    )
                            down_aiv_store_ub = down_aiv_out_ub
                        tla.set_flag(down_aiv_done)
                        tla.wait_flag(down_aiv_done)
                        tla.copy(down_aiv_gm_c_tile, down_aiv_store_ub)
                        tla.pipe_barrier(tla.pipes.ALL)
        tla.pipe_barrier(tla.pipes.ALL)
