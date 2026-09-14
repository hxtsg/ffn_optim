"""Single mixed FFN kernel, composed from pinned CATLASS device bodies.

MatMul loops are extracted, not reimplemented. No donor Host code is imported.
Generated derivative code retains the CATLASS CANN Open Software License 2.0.
CPU callers may inspect generate_source() without importing the DSL or NPU.
"""
import ast
import copy
import hashlib
import importlib.util
import linecache
import sys

from ops.host.dispatch import EXAMPLES, SOURCES, Selection, verify_dependency


def _nodes(source):
    return ast.parse(source).body


class _Namespace(ast.NodeTransformer):
    def __init__(self, function, prefix, arguments):
        locals_ = {node.id for node in ast.walk(function)
                   if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.names = {name: prefix + name for name in locals_}
        self.names.update(arguments)
        self.prefix = prefix

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(self.names.get(node.id, node.id), node.ctx), node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr in ('flag', 'cross_flag')
                and node.args and isinstance(node.args[0], ast.Constant)):
            node.args[0].value = self.prefix + node.args[0].value
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # Donor FP16 SK reduction floors; use nearest instead for FFN output.
        if node.attr == 'CAST_FLOOR':
            node.attr = 'CAST_ROUND'
        return node


def _extract(kind, prefix, arguments):
    tree = ast.parse(SOURCES[kind].read_text())
    name = 'basic_mmad_kernel' if kind == 'basic' else 'streamk_mmad_kernel'
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f'Donor anchor changed: {name}')
    function = _Namespace(matches[0], prefix, arguments).visit(copy.deepcopy(matches[0]))
    declarations, regions = [], {}
    for node in function.body:
        if isinstance(node, ast.With):
            context = node.items[0].context_expr
            if not isinstance(context, ast.Call) or not isinstance(context.func, ast.Attribute):
                raise RuntimeError('Unexpected donor region')
            regions[context.func.attr] = node.body
        else:
            declarations.append(node)
    if set(regions) != ({'cube'} if kind == 'basic' else {'cube', 'vector'}):
        raise RuntimeError('Unexpected donor region topology')
    return declarations, regions


DECLARATIONS = '''
tile_available = tla.cross_flag("ffn_tile_available", mode=4)
tile_ready = tla.cross_flag("ffn_tile_ready", mode=4)
up_ready = tla.cross_flag("ffn_up_ready", mode=2)
all_aic = tla.cross_flag("ffn_all_aic", mode=0)
down_release = tla.cross_flag("ffn_down_release", mode=2)
gelu_done = tla.flag("ffn_gelu_done", tla.arch.VECTOR, tla.arch.MTE3)
acc_ptr = tla.allocate(4096, tla.Float32, tla.AddressSpace.ub, 256)
out_ptr = tla.allocate(4096, hidden.ptr.dtype, tla.AddressSpace.ub, 256)
ub_layout = tla.make_layout(tla.make_shape(64, 64), tla.make_stride(64, 1), layoutTag=tla.arch.RowMajor)
acc_ub = tla.make_tensor(acc_ptr, ub_layout)
out_ub = tla.make_tensor(out_ptr, ub_layout)
acc_1d = tla.make_tensor(acc_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
out_1d = tla.make_tensor(out_ptr, tla.make_layout(tla.make_shape(4096), tla.make_stride(1)))
cast_params = tla.params.CastParams(reg_slot=tla.params.RegSlot.ZERO, sat_mode=tla.params.SatMode.NOSAT, round_mode=tla.params.RoundMode.CAST_ROUND)
store_params = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
'''

UP_COPY = '''
tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
tla.copy(acc_ub, up_l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11, l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0))
tla.cross_core_set_flag(tile_ready, tla.arch.FIX, aiv_id=0)
'''

CUBE_BARRIER = '''
tla.cross_core_wait_flag(tile_available, tla.arch.FIX, aiv_id=0)
tla.pipe_barrier(tla.pipes.ALL)
tla.cross_core_wait_flag(up_ready, tla.arch.MTE2)
tla.cross_core_set_flag(all_aic, tla.arch.MTE2)
tla.cross_core_wait_flag(all_aic, tla.arch.MTE2)
tla.cross_core_set_flag(down_release, tla.arch.MTE2)
'''

# All AIVs, including the idle second AIV and no-tile cores, reach the barrier.
VECTOR_UP = '''
if tla.arch.sub_block_idx() == 0:
    tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
    for vector_tile in tla.range(tla.arch.block_idx(), up_total_blocks, tla.arch.block_num()):
        tla.cross_core_wait_flag(tile_ready, tla.arch.VECTOR, aiv_id=0)
        with tla.vec.func(mode="simd"):
            mask32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            mask16 = tla.create_mask(pattern=tla.mask.ALL, dtype=hidden.ptr.dtype)
            for chunk in tla.range(0, 64, 1):
                src = tla.tile_view(acc_1d, tla.make_shape(64), tla.make_coord(chunk))
                dst = tla.tile_view(out_1d, tla.make_shape(64), tla.make_coord(chunk))
                z = src.load()
                t = tla.mul(z, 0.70710678118654752, mask=mask32)
                t = tla.max(t, -3.92, mask=mask32)
                t = tla.min(t, 3.92, mask=mask32)
                t2 = tla.mul(t, t, mask=mask32)
                p = tla.add(tla.mul(t2, 0.053443748819, mask=mask32), 7.5517016694, mask=mask32)
                p = tla.add(tla.mul(p, t2, mask=mask32), 101.62808918, mask=mask32)
                p = tla.add(tla.mul(p, t2, mask=mask32), 1393.8061484, mask=mask32)
                p = tla.add(tla.mul(p, t2, mask=mask32), 5063.7915060, mask=mask32)
                p = tla.add(tla.mul(p, t2, mask=mask32), 29639.384698, mask=mask32)
                p = tla.mul(p, t, mask=mask32)
                q = tla.add(t2, 31.212858877, mask=mask32)
                q = tla.add(tla.mul(q, t2, mask=mask32), 398.56963806, mask=mask32)
                q = tla.add(tla.mul(q, t2, mask=mask32), 3023.1248150, mask=mask32)
                q = tla.add(tla.mul(q, t2, mask=mask32), 13243.365831, mask=mask32)
                q = tla.add(tla.mul(q, t2, mask=mask32), 26267.224157, mask=mask32)
                y = tla.mul(tla.mul(z, 0.5, mask=mask32), tla.add(tla.div(p, q, mask=mask32), 1.0, mask=mask32), mask=mask32)
                dst.store(y.to(hidden.ptr.dtype, cast_params, mask32), store_params, mask=mask16)
        tla.set_flag(gelu_done)
        tla.wait_flag(gelu_done)
        hidden_tile = tla.tile_view(hidden, tla.make_shape(64, 64), tla.make_coord(vector_tile // up_grid_n, vector_tile % up_grid_n))
        tla.copy(hidden_tile, out_ub)
        tla.cross_core_set_flag(tile_available, tla.arch.MTE3, aiv_id=0)
tla.pipe_barrier(tla.pipes.ALL)
tla.cross_core_set_flag(up_ready, tla.arch.MTE3)
tla.cross_core_wait_flag(down_release, tla.arch.MTE2)
tla.pipe_barrier(tla.pipes.ALL)
'''


class _UpOutput(ast.NodeTransformer):
    replacements = 0

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'copy' and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name) and call.args[0].id == 'up_gm_c_by_core'
                and isinstance(call.args[1], ast.Name) and call.args[1].id == 'up_l0_c'):
            self.replacements += 1
            return _nodes(UP_COPY)
        return self.generic_visit(node)


def generate_source(down_impl='basic'):
    """Return auditable source and provenance; never import or launch device code."""
    Selection(down_impl=down_impl).require_available()
    hashes = verify_dependency()
    up_decl, up = _extract('basic', 'up_', {
        'gm_a': 'x', 'gm_b': 'weight1', 'gm_c': 'hidden', '_tiling': 'up_tiling'})
    down_decl, down = _extract(down_impl, 'down_', {
        'gm_a': 'hidden', 'gm_b': 'weight2', 'gm_c': 'output',
        'gm_workspace': 'workspace', '_tiling': 'down_tiling', '_swizzle': 'swizzle'})
    adaptation = _UpOutput()
    up_module = adaptation.visit(ast.Module(body=up['cube'], type_ignores=[]))
    if adaptation.replacements != 1:
        raise RuntimeError('Expected exactly one Basic output-copy anchor')
    tree = ast.parse('''
@tla.kernel
def ffn_kernel(x: tla.Tensor, weight1: tla.Tensor, weight2: tla.Tensor,
               hidden: tla.Tensor, output: tla.Tensor, workspace: tla.Tensor,
               up_tiling: TilingParams, down_tiling: TilingParams,
               swizzle: SwizzleParams, block_dim: tla.Constexpr[int],
               hf32_mode: tla.Constexpr[tla.params.HF32Mode]):
    pass
''')
    cube = _nodes('with tla.cube():\n    pass')[0]
    vector = _nodes('with tla.vector():\n    pass')[0]
    cube.body = up_module.body + _nodes(CUBE_BARRIER) + down['cube']
    vector.body = _nodes(VECTOR_UP) + down.get('vector', [])
    tree.body[0].body = up_decl + down_decl + _nodes(DECLARATIONS) + [cube, vector]
    ast.fix_missing_locations(tree)
    license_header = '\n'.join(SOURCES['basic'].read_text().splitlines()[:10]) + '\n'
    source = (license_header + '# Derived from CATLASS v2.0.0 device samples; CANN Open Software License 2.0\n'
              '# MatMul source provenance is recorded in the run manifest\n' + ast.unparse(tree) + '\n')
    compile(source, '<ffn-static-check>', 'exec')  # syntax only, no execution
    return source, {'donor_sha256': hashes, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'down_impl': down_impl, 'up_output_copy_adaptations': adaptation.replacements}


def compile_kernel(tensors, plan):
    """Explicit device path only. Importing this module does not call this function."""
    source, provenance = generate_source(plan.selection.down_impl)
    import catlass.tla as tla
    path = EXAMPLES / 'common/params.py'
    spec = importlib.util.spec_from_file_location('_ffn_catlass_params', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = {'__name__': 'ops.kernel.ffn', 'tla': tla, 'TilingParams': module.TilingParams,
                 'SwizzleParams': module.SwizzleParams, 'AIV_TILE_M': 16,
                 'AIV_SUB_BLOCK_NUM': 2, 'AIV_REG_M': 1, 'AIV_REG_N': 64}
    filename = '<ffn_' + provenance['source_sha256'] + '.py>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)
    from dataclasses import asdict
    artifact = tla.compile(namespace['ffn_kernel'], *tensors,
                           module.TilingParams(**asdict(plan.up)),
                           module.TilingParams(**asdict(plan.down)), module.SwizzleParams(),
                           plan.block_num, tla.params.HF32Mode.HF32_DISABLE,
                           options='--npu-arch 3510')
    return artifact, provenance
