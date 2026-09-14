# FFN V1

按`spec/v1_spec.md`实现的Ascend950实验工程，当前已完成CPU及静态检查，尚未验证DSL编译、NPU正确性、同步或性能

$$
Y=\operatorname{GELU}(XW_1^\mathsf T)W_2^\mathsf T
$$

## 范围

- 无Bias、仅GELU、FP16/BF16，显式`layout="linear"`
- X支持2～8维，最后一维为K；W1为`[H,K]`，W2为`[N,H]`
- 非连续输入在入口连续化，M/H/N补零到64，K补零到16，输出裁剪并恢复前导维
- 上投影Basic，下投影可手动选择Basic或Stream-K
- A/B全载保留配置名，明确返回`unavailable`，不会退回Basic
- 外层搜索枚举指定候选集合，先验精度再计时，只在实测通过的组合中选最优
- FFN主体只有一次artifact下发，入口补零、连续化及数据生成是额外辅助任务
- 仅推理，不支持autograd、Bias、Canonical或任意激活参数；布局扩展仍待确认

## 项目整体设计

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 算子Host | `ops/api.py`、`ops/host/` | 参数检查、连续化与补零、策略、核数、workspace和编译下发 |
| 算子Kernel | `ops/kernel/ffn.py` | 同文件组织cube/vector、GELU、同步和MatMul组合 |
| 精度验证 | `validation/run.py`、`reference.py`、`test_ffn.py` | torch参考、统一误差判据和回归测试 |
| 性能采集 | `performance/run.py`、`tune.py`、`collect.py` | 手动配置/组合搜索、预热、设备Event和独立profiler采集 |
| 性能分析 | `performance/analyze.py` | 汇总证据、记录未确认瓶颈和优化建议 |
| 公共配置 | `common/`、`configs/v1_cases.json` | case、固定随机输入和版本快照 |

`test_ffn.py`同时承载误差函数和测试，不另设metrics.py；不创建sync.py、gelu.py或项目内matmul目录

## 依赖与复用方式

CATLASS子模块固定v2.0.0，提交`769cd40a8716b28650b6bebb08db4834eea4462f`

```bash
git submodule update --init 3rdparty/catlass
python -m pip install -e .
```

设备环境另外需要匹配的torch-npu、CANN和可用CATLASS-DSL编译/runtime组件，安装方式参考子模块`python/tla_dsl/README.md`，不通过本工程自动安装或改动这些组件
CPU检查仅需Python≥3.10和torch；默认入口不导入torch_npu或catlass

MatMul直接取自子模块`python/tla_dsl/examples/end_to_end/`

- Basic：`basic_mmad/basic_matmul.py`的`basic_mmad_kernel`
- Stream-K：`basic_mmad_streamk/basic_mmad_streamk.py`的`streamk_mmad_kernel`
- mixed组织参考：`basic_mixed/basic_mixed_ub2l1.py`
- L0C→UB单AIV通路参考：`basic_mixed/basic_mixed_fixpipe_nz2dn.py`

上述样例是独立Kernel，不是可直接嵌入的设备函数，所以`ffn.py`在CPU侧读取其AST，提取设备区域并重命名局部变量和flag，再生成唯一FFN入口
不调用样例Host函数，不复制维护一套MatMul算法，不修改子模块

适配仅包括：参数映射、局部缓冲/flag隔离、将上投影最终GM写回替换为L0C→UB、增加GELU和阶段同步，以及将Stream-K归约FP16输出的FLOOR转换改为ROUND
生成时检查固定提交、tracked修改和输出写回锚点，任一不匹配即失败；生成源码与源文件SHA256写入结果目录供审查
生成代码保留CATLASS版权头，相关派生部分遵循子模块`LICENSE`中的CANN Open Software License Agreement Version 2.0

## Kernel计算与同步

上投影复用Basic的GM→L1→L0供数和MMAD，FP32结果从L0C写UB
每个AIC仅使用配对的AIV0执行GELU，转为输入dtype后写GM hidden
mode4的available/ready握手保护逐tile UB复用，最后消费完毕后再进入阶段屏障；AIV1不参与GELU，但必须参与阶段同步

阶段间顺序为：AIV完成hidden写回→mode2报到→AIC mode0汇合→mode2放行→读取hidden执行下投影
无上投影任务的核也执行同样的阶段屏障，不以PIPE_ALL代替跨核同步
Stream-K下投影保留样例的K分片、每核两块FP32 scratch、AIC/AIV同步及AIV归约，仍在同一次下发内

当前保守Tiling为L1=`64×64×128`、L0=`64×64×32`，up/down使用相同block_num，默认8且执行前不允许超过物理AIC数
Stream-K要求存在尾轮tile且H方向至少两块L1 K块，否则显式拒绝
这不是MindIE-SD的自动Tiling复刻，也不是性能最优参数；搜索范围当前是实现组合，不包含全部Tiling参数

workspace为Basic的未使用占位或Stream-K的`[2*64*block_num,64]`FP32 scratch，hidden为`[padM,padH]`输入dtype
各缓冲独立分配，避免手动拼接GM区间的偏移对齐问题；补零会增加实际计算量，须纳入性能分析
返回Tensor可能是带padding行步长的非连续view；`PreparedFFN.run()`重复使用同一输出缓冲，若需保存历史结果需自行clone，并在准备时的同一stream串行使用

## 精度

默认golden用torch全FP32的MatMul→erf GELU→MatMul后转回输入dtype
同时提供分阶段hidden舍入参考和基线有理式GELU，辅助定位精度差异
通过标准为shape/dtype匹配、双方有限且`mean(abs(out-golden)/(abs(golden)+1e-3)) < 0.05`，同时报告最大绝对误差与RMS

GELU有理式系数对应MindIE-SD提交`4cb292aec256a3bb3c0deea51c5cf55426b1f0e7`的`csrc/ops/eagle_ffn/op_kernel/3rd/mat_mul_v3/arch35/cmct/epilogue/fusion/fusion_regbase_act.h`
DSL当前使用CAST_ROUND，不将其声称为基线RINT的逐位等价实现；归约顺序也可能不同

## 本轮允许执行的检查

以下命令不运行NPU

```bash
python -m unittest validation.test_ffn -v
python -m validation.run --dry-run
python -m performance.run --mode search --dry-run
python -m compileall -q common ops validation performance
git diff --check
```

dry-run只生成参数计划、源码和适用性结果，不编译DSL、不创建NPU Tensor、不填性能数字、不生成虚假的最优配置
配置文件目前只有3个smoke case，不是case全集

本轮检查记录：2026-09-12，Python3.12.14、torch2.14.0，17项CPU测试通过，20个Python文件AST及新增文件空白检查通过
手动验证dry-run及12组合搜索dry-run通过，确认未导入catlass或torch_npu；设计文档、ref及CATLASS子模块未修改
测试环境缺少NumPy产生torch初始化警告，本轮测试未使用NumPy，不影响上述通过结果

## 后续设备执行入口

以下命令仅在另行授权NPU执行后使用，本轮未运行

```bash
# 单次手动配置，可通过--case-id筛选case
python -m performance.run --execute-npu --device npu:0 --down-impl basic

# 组合搜索；全载选项记录unavailable，不伪造结果
python -m performance.run --mode search --execute-npu --device npu:0 \
  --down-candidates basic full_load_a full_load_b streamk --warmup 5 --repeats 20

# 只做设备正确性和重复调用检查
python -m validation.run --execute-npu --down-impl basic

# Event计时之外追加独立profiler采集
python -m performance.run --execute-npu --down-impl streamk --profile
```

完整接口示例

```python
from ops import ffn, prepare_ffn

y = ffn(x, weight1, weight2, down_impl="basic", block_num=8)
# 性能采集使用PreparedFFN，避免把输入处理和JIT计入FFN主体
prepared = prepare_ffn(x, weight1, weight2, down_impl="streamk", block_num=8)
y = prepared.run()
```

`configs/v1_cases.json`记录case_id、输入/权重shape、dtype、layout、seed和noncontiguous
手动配置使用`--up-impl/--down-impl`；搜索模式使用`--up-candidates/--down-candidates`的笛卡尔积，固定输入串行评估

## 结果与验收边界

每次运行创建`results/<run_id>/`，含manifest.json、accuracy.json、performance.csv、best_configs.json、生成源码、profiler/和analysis.md，不覆盖历史运行
搜索逐候选checkpoint；编译失败、精度失败及不适用均保留原因；设备准备/运行异常中止搜索，避免在可能异常的设备上下文继续下发

- `compile_wall_ms`单列编译；`auxiliary_device_interval_ms`记录连续化、分配和补零所在设备区间，不冒充单个辅助task时长
- 不带profiler的Event中位数用于排名，`task_duration_us`只从匹配FFN的设备trace事件读取，不混同Host时间
- profiler接口可用时请求PipeUtilization；从op_summary中提取明确标注的FFN task时长和流水ratio/utilization字段，并保留原始列名单位
- 原始profiler产物保留；接口或导出字段未提供流水利用率时输出不可用，不填0，也不推算成实测值
- 缺少阶段级证据时，GELU、GM流量、同步和SK归约仅列为待验证瓶颈
- 本轮只验Python语法、源码结构和CPU数学/工程逻辑，没有执行NPU编译、正确性、同步重复调用或性能测试
- 首次设备验收须检查编译资源、mode4及mode0/2 lowering、无任务核参与、连续重复调用和非对齐结果，再开始性能比较
