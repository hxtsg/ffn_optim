"""Explicit NPU execution API; CPU/static import is safe."""
from dataclasses import dataclass
from time import perf_counter
from ops.host.dispatch import Selection, verify_dependency
from ops.host.inputs import validate_tensors, prepare_storage
from ops.host.tiling import make_plan


class KernelCompilationError(RuntimeError):
    """功能：封装编译阶段失败，使外层搜索可记录失败并尝试其他候选

    输入：编译异常的类型及消息字符串，原异常通过异常链保留
    输出：可捕获的RuntimeError异常；此时尚未下发FFN主体，但可能已执行输入准备
    """


@dataclass
class PreparedFFN:
    """功能：持有已准备的输入、缓冲和编译产物，支持重复下发同一FFN

    输入：由prepare_ffn创建，plan须与storage、tensors和artifact一致
    storage依次为X[mp,kp]、W1[hp,kp]、W2[np,hp]、hidden[mp,hp]、输出[mp,np]及scratch
    前五项为同一NPU上的FP16/BF16 Tensor，scratch为FP32，tensors为对应DSL视图
    provenance/preparation分别记录来源及准备耗时，stream为准备时的执行流
    输出：run()每次下发一个FFN主体Kernel，返回原前导维加N的同dtype输出view
    输出与内部缓冲共享存储，下次run()会覆盖；须在原stream串行调用，不保证返回时设备已完成
    """

    plan: object
    storage: tuple
    tensors: tuple
    artifact: object
    provenance: dict
    preparation: dict
    stream: object

    def run(self):
        """One FFN launch. Returned view is overwritten by the next run."""
        import torch
        with torch.npu.device(self.storage[0].device):
            if torch.npu.current_stream() != self.stream:
                raise RuntimeError('PreparedFFN must run on its preparation stream; concurrent reuse is unsupported')
            self.artifact(*self.tensors, block_num=self.plan.block_num)
        p = self.plan.problem
        return self.storage[4][:p.m, :p.n].view(p.output_shape)


def prepare_ffn(x, weight1, weight2, *, up_impl='basic', down_impl='basic',
                block_num=8, layout='linear'):
    """Prepare padded storage and compile, but do not launch FFN.

    Padding/continuous copies are auxiliary device operations, outside FFN timing.
    Call only when device execution has explicitly been authorized.
    """
    p = validate_tensors(x, weight1, weight2, layout=layout, require_npu=True)
    plan = make_plan(p, Selection(up_impl, down_impl), block_num)
    verify_dependency()
    import torch
    import torch_npu  # noqa: F401 - register NPU only on this explicit path
    props = torch.npu.get_device_properties(x.device)
    if '950' not in torch.npu.get_device_name(x.device):
        raise RuntimeError('This implementation targets Ascend950 only')
    if block_num > int(props.cube_core_num):
        raise ValueError('Global barrier requires block_num <= physical AIC count')
    if int(props.vector_core_num) != 2 * int(props.cube_core_num):
        raise RuntimeError('Expected two AIVs per AIC')
    from catlass.tla.runtime import from_dlpack
    import catlass.tla as tla
    from ops.host.dispatch import compile_kernel
    with torch.npu.device(x.device):
        stream = torch.npu.current_stream()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        storage = prepare_storage(x, weight1, weight2, plan)
        end.record()
        end.synchronize()
        auxiliary_ms = start.elapsed_time(end)
        layouts = ('row', 'col', 'col', 'row', 'row', 'row')
        tensors = tuple(from_dlpack(buf, layout_tag=(tla.arch.RowMajor if layout == 'row'
                                                    else tla.arch.ColumnMajor)).mark_layout_dynamic()
                        for buf, layout in zip(storage, layouts))
        before = perf_counter()
        try:
            artifact, provenance = compile_kernel(tensors, plan)
        except Exception as exc:
            raise KernelCompilationError(f'{type(exc).__name__}: {exc}') from exc
        preparation = {'auxiliary_device_interval_ms': auxiliary_ms,
                       'compile_wall_ms': (perf_counter() - before) * 1000}
    return PreparedFFN(plan, storage, tensors, artifact, provenance, preparation, stream)


def ffn(x, weight1, weight2, **configuration):
    """Y=GELU(X@weight1.T)@weight2.T; no bias, no autograd."""
    return prepare_ffn(x, weight1, weight2, **configuration).run()
