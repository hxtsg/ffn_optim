# V1需求与设计

## 1. 目标与范围

基于CATLASS DSL，在Ascend950上实现无Bias、仅GELU的融合FFN

| 项目 | V1约束 |
| --- | --- |
| 上采样 | BasicMatmul＋GELU，保留后续接入其他MatMul实现的扩展点 |
| 下采样 | 按原大纲保留Basic、A全载、B全载、Stream-K候选，不要求每种实现适用于所有shape |
| MatMul来源 | 复用CATLASS v2.0.0的DSL end_to_end已有实现，本项目不另写MatMul实现，缺失入口显式记录待确认 |
| Kernel下发 | FFN主体仅下发一次，以mode0＋mode2跨核flag协议实现阶段间全核同步 |
| 选择方式 | 单次调用可手动指定MatMul实现，外层调优遍历候选组合，按实测性能选择每个case的最优组合 |
| 数据类型 | 与基线非量化路径一致，输入和权重同为FP16或BF16，hidden和输出使用输入dtype |
| 测试设备 | 默认`npu:0`，用户已确认CANN和DSL安装完成，连接方式及实际版本待记录 |
| 暂不包含 | Bias、SiLU、SwiGLU、量化、MoE，以及上采样全载或Stream-K实现 |
| 输入兼容 | 入口连续化、2～8维输入和非对齐尺寸，权重布局范围待确认 |
| 待补充 | case全集、性能对照和目标、权重布局、同步协议的NPU验证 |

此处的上、下采样分别对应原大纲中的上、下采样，不涉及空间插值

## 项目整体设计

1. 算子逻辑：包括Host逻辑和Kernel逻辑
2. 精度验证：执行torch计算，比较结果，验证精度通过
3. 性能采集分析：针对当前版本实现，运行配置的全部case，支持外层遍历MatMul组合调优，获取性能结果及最优配置

   - 3.1 性能采集：包括算子执行的task duration和pipeline利用率等
   - 3.2 性能分析：分析算子执行瓶颈，给出后续优化思路

### 项目文件结构

按上述三部分组织项目，以下为规划结构，不表示文件已经实现；保留现有`spec/`、`ref/`、基线分析文档及CATLASS子模块
Python包所需的`__init__.py`在树中省略

```text
ffn_optim/
├── README.md                         # 环境、入口命令和结果查看方式
├── pyproject.toml                    # 项目安装及Python依赖声明
├── spec/
│   └── v1_spec.md                    # 本需求与设计文档
├── 3rdparty/
│   └── catlass/                      # 固定v2.0.0，复用python/tla_dsl/examples/end_to_end实现
├── ref/
│   └── triton_impl.py                # 已有多Kernel参考，不作为V1执行后端
├── MindIE-SD EagleFFN算子实现分析.md  # 已有基线实现分析
├── configs/
│   └── v1_cases.json                 # case清单，全集待补充
├── common/
│   ├── cases.py                      # 读取case、生成可复现输入，供验证和采集共用
│   └── environment.py                # 默认npu:0，记录设备及软件版本
├── ops/                              # 1. 算子逻辑
│   ├── api.py                        # FFN公开调用接口，连接Host处理与Kernel下发
│   ├── host/
│   │   ├── inputs.py                 # 参数检查、连续化、2～8维展平及输出shape恢复
│   │   ├── dispatch.py               # 映射CATLASS已有实现及适用性检查，不自动切换路径
│   │   └── tiling.py                 # 适配已有MatMul参数，统一up/down核数和workspace
│   └── kernel/
│       └── ffn.py                    # 单个FFN入口，同文件组织cube/vector、GELU和跨核同步
├── validation/                       # 2. 精度验证
│   ├── run.py                        # 验证入口，逐case调用算子并生成精度报告
│   ├── reference.py                  # torch全FP32参考和分阶段舍入参考
│   └── test_ffn.py                   # shape/dtype/有限值检查、误差判定及输入边界、非法配置、同步回归测试
├── performance/                      # 3. 性能采集分析
│   ├── run.py                        # 总入口，支持单次手动配置和外层搜索调优两种模式
│   ├── tune.py                       # 遍历候选组合，先验证再计时，为每个case选择最优配置
│   ├── collect.py                    # 3.1 预热、设备计时、profiler采集及指标提取
│   └── analyze.py                    # 3.2 汇总结果、分析瓶颈、生成优化建议报告
└── results/                          # 运行产物，按run_id隔离，不覆盖历史结果
    └── <run_id>/
        ├── manifest.json            # case、随机种子、路径配置、版本和计时参数快照
        ├── accuracy.json            # 每个case的精度、错误或跳过原因
        ├── performance.csv          # 每个case/路径的时长及可用流水指标
        ├── best_configs.json        # 每个case的最优组合、性能、搜索范围及完成状态
        ├── profiler/                # profiler原始采集产物
        └── analysis.md              # 瓶颈证据、未确认项及后续优化思路
```

### 三部分如何衔接

- **算子逻辑**：`ops/api.py`调用Host检查和参数适配，再下发`ops/kernel/ffn.py`；MatMul复用CATLASS已有实现，GELU、hidden转换及跨核同步与cube/vector逻辑一起写在`ffn.py`中，不另建`gelu.py`、`sync.py`或项目内`matmul/`
- **精度验证**：`validation/run.py`对同一组输入分别调用FFN和torch参考，复用`test_ffn.py`中的检查、误差计算及通过判定函数；该文件同时承载输入边界、非法配置和同步重复调用的回归测试，通过标准见第7节
- **性能采集分析**：`performance/run.py`支持手动模式和搜索模式，搜索模式由`tune.py`生成组合，两种模式都先验精度、再由`collect.py`计时，最后由`analyze.py`输出报告，采集与分析要求见第8节

`performance/run.py`承载单次配置和搜索空间：手动模式按`case_id`指定`up_impl`和`down_impl`，搜索模式遍历两者候选集合的笛卡尔积
每次评估都将一个确定配置传给算子和验证逻辑，单次调用不擅自换路径；最优组合由外层`tune.py`依据实测结果选择
case文件记录`case_id`、输入shape、权重shape、dtype、布局和连续性构造方式，布局取值在范围确认后补齐
验证失败或配置不适用时保留原因，不生成正常性能成绩；采集工具不提供的流水指标标记为不可用，不填0或推算成实测值
`analysis.md`中的瓶颈结论关联实际指标或trace，证据不足时标为待验证；结果目录为运行产物，不作为源码提交

### CATLASS实现复用边界

以子模块中的`python/tla_dsl/examples/end_to_end/`为来源，采用`basic_mixed/basic_mixed_ub2l1.py`在同一个`@tla.kernel`内组织`with tla.cube()`和`with tla.vector()`的方式[^D5]
该mixed样例用于参考代码组织和跨核协作，不将它的FP32、32×32尺寸或UB→L1数据通路当作FFN的功能约束，FFN阶段间仍按本设计通过GM传递hidden

| 用途 | 已定位来源，相对于上述end_to_end目录 | 接入边界 |
| --- | --- | --- |
| mixed组织方式 | `basic_mixed/basic_mixed_ub2l1.py` | cube/vector、flag和缓冲在同一Kernel中组织 |
| BasicMatMul | `basic_mmad/basic_matmul.py`中的`basic_mmad_kernel` | 复用已有设备计算及Tiling参数，适配FFN的输入输出与融合边界 |
| Stream-K MatMul | `basic_mmad_streamk/basic_mmad_streamk.py`中的`streamk_mmad_kernel` | 复用已有K分片与归约逻辑，归约保持在同一次FFN下发内 |
| A/B全载MatMul | 固定标签的DSL end_to_end中暂未定位到对应样例 | 保留需求，待确认已有实现入口，不在本项目重写或冒充已接入 |

已定位的Basic和Stream-K样例以独立`@tla.kernel`入口组织，并带有各自的Host编译、下发逻辑，不等于已经提供可直接嵌入FFN的设备函数接口[^D5]
接入时需要核验已有设备逻辑的组合方式，不能直接调用两个样例的`run()`或执行两个独立artifact来冒充单Kernel融合
不在本项目复制维护一套独立MatMul算法，也不未经确认修改子模块；若固定版本缺少所需的可组合入口，先确认依赖侧适配方式

## 2. 数学表达式与精度

采用Linear权重布局时，完整计算为：

$$
Y=\operatorname{GELU}(XW_1^\mathsf T)W_2^\mathsf T
$$

其中X为`[M,K]`，W1为`[H,K]`，W2为`[N,H]`，输出Y为`[M,N]`

GELU采用基线的erf定义：

$$
\operatorname{GELU}(z)=\frac{z}{2}\left[1+\operatorname{erf}\left(\frac{z}{\sqrt2}\right)\right]
$$

设备计算对齐基线的精度边界：FP32累加并执行GELU，hidden转换为输入dtype后写GM，下采样使用FP32累加，最终输出转回输入dtype
基线设备GELU使用erf有理式近似，移植时参考该实现，不将tanh近似作为等价替换，也不要求不同归约顺序逐位一致[^B6]

## 3. 依赖与版本

- CATLASS子模块：`3rdparty/catlass`
- 远端：`https://gitcode.com/cann/catlass.git`
- 固定标签：`v2.0.0`
- 固定提交：`769cd40a8716b28650b6bebb08db4834eea4462f`
- DSL源码入口：`3rdparty/catlass/python/tla_dsl`
- FFN基线提交：`4cb292aec256a3bb3c0deea51c5cf55426b1f0e7`

子模块以gitlink固定提交，不跟随远端分支更新，初始化命令为：

```bash
git submodule update --init 3rdparty/catlass
```

用户已确认执行环境安装了CANN和DSL，本次不重复安装或构建，后续核验已安装DSL是否对应固定标签
DSL的嵌套依赖和构建环境按固定版本文档配置
该版本DSL README声明最低CANN版本为9.1.0，实际执行环境的CANN、驱动、torch、torch_npu和DSL构建版本须记录到测试报告[^D1]

## 4. Host设计

### 4.1 配置与调用

选择分为两层：单次调用接收确定的上、下采样实现；外层调优负责生成并遍历组合，不在一次FFN调用内部搜索
以下为单次调用的拟定配置形式，不是已实现API：

```python
config = {
    "device": "npu:0",
    "up_impl": "basic",
    "down_impl": "basic",  # basic / full_load_a / full_load_b / streamk
}
```

每个case的每个确定配置按以下顺序执行：

1. 检查shape、dtype、布局及手动选择的实现是否合法
2. 按选定实现分别计算up/down的分块、核数、缓冲和workspace
3. 组合为一个FFN编译变体，准备hidden及必要的归约、同步空间
4. 下发一次FFN主体，输出Y

不适用的手动配置应明确报告原因，不静默改用其他实现
搜索模式将不适用组合记录为跳过、精度失败组合记录为失败，再继续评估其他组合，不把换路径后的结果归给原配置
不同case可以选择不同编译变体，但每次FFN主体仍只有一次下发

外层搜索空间在性能入口配置，V1默认遍历以下实现组合：

```python
search_space = {
    "up_impl": ["basic"],
    "down_impl": ["basic", "full_load_a", "full_load_b", "streamk"],
}
```

V1每个case有4种待检查组合，只评测适用且精度通过的组合；未来接入其他上采样实现时扩展候选集合，遍历机制保持不变
一次调优会执行多次FFN调用，但每次FFN主体仍只下发一个Kernel

### 4.2 组件边界

| 组件 | 职责 |
| --- | --- |
| 输入处理 | 对X、W1、W2入口连续化，检查2～8维输入并生成二维逻辑shape |
| MatMul实现注册 | 将实现名称映射到CATLASS已有实现及参数接口，缺失或无法组合的入口明确报告 |
| 分阶段Tiling | 复用并适配CATLASS已有分块和调度参数，为up/down分别准备配置，不另建MatMul算法实现 |
| FFN组合 | 在`ffn.py`中统一cube/vector逻辑、GELU、launch核数、workspace和阶段同步 |
| 性能入口与外层调优 | 管理case、手动配置及搜索空间，遍历候选组合，记录全部评估结果和每个case的最优配置 |

采用一个统一的launch核数，两阶段组件必须适配该核数，不得内部再次launch
无计算任务的核也必须遵守全核同步协议，不能提前返回导致其他核等待
后续扩展上采样实现时接入CATLASS已有实现，不修改FFN的数学定义和单次下发约束

## 5. Kernel设计

设备逻辑统一组织在`ops/kernel/ffn.py`，参照mixed样例在同一FFN Kernel内书写cube/vector区域
GELU及hidden转换放在vector逻辑中，mode0/mode2阶段同步也写在该文件中，不拆分独立实现文件
MatMul来源及接入边界见“项目整体设计”，以下描述组合后的计算流程，不要求本项目新写MatMul算法

V1主体执行顺序为：

```text
Basic上采样：X × W1ᵀ
    → AIV执行GELU并转换dtype
    → hidden写GM
    → mode2报到 → mode0全AIC汇合 → mode2放行
    → 选定的下采样组件读取hidden并计算hidden × W2ᵀ
    → Y写GM
```

hidden为`[M,H]`，FP16/BF16下有效大小为`2*M*H`字节，生命周期从上采样写入到下采样读完
Stream-K下采样还需要FP32局部和及归约逻辑，归约也必须在同一次FFN下发中完成
A全载指hidden加载到各参与核的L1常驻，B全载指W2常驻，两者均不是直接继承上采样的片上数据

各workspace区域按实际DMA、归约和同步访问要求对齐，使用足够宽的整数计算大小和偏移，并检查溢出
具体Tiling映射、事件编号、buffer容量及workspace分区按所复用的CATLASS实现补充，不能把MindIE-SD基线参数原样当作DSL可用配置

X、W1、W2位于设备GM，Y也留在设备GM，测试程序可在算子调用前后执行Host↔Device拷贝，不把拷贝固定写入设备Kernel

### 5.1 用mode0＋mode2实现阶段间全核同步

V1采用跨核flag组合，满足基线`SyncAll<false>()`在此处的阶段依赖：任何下采样读取开始前，全部上采样hidden已写入GM
这是本FFN场景的等价同步协议，不声称复刻SyncAll的所有重载和使用场景[^B5]

Ascend950的1AIC＋2AIV模式下，mode0用于同类核间汇合，mode2用于一个AIC与其两个AIV之间的同步[^D3]

| 步骤 | AIC侧 | 对应的两个AIV侧 |
| --- | --- | --- |
| 1. 本核收尾 | 等待本核上采样流水完成 | 等待GELU及hidden的GM写回完成 |
| 2. mode2报到 | 等待两个AIV的`upReady`信号 | 各发送一次`upReady` |
| 3. mode0汇合 | 对`allAicArrived`执行set＋wait，等待所有参与AIC到达 | 等待放行 |
| 4. mode2放行 | 向两个AIV发送`downRelease`，随后进入下采样 | 收到`downRelease`后进入下采样阶段 |

因每个AIC报到前已等待其两个AIV，所有AIC汇合就保证了所有参与AIC/AIV的上采样已完成
AIV侧也必须等待放行，尤其Stream-K下采样仍需要AIV参与归约，不能提前复用上一阶段缓冲

DSL使用现有`cross_flag`、`cross_core_set_flag`和`cross_core_wait_flag`，无需新增同名SyncAll接口[^D2]
以下为嵌入FFN Kernel的同步片段设计，尚未编译或执行，不是独立可运行Kernel：

```python
# 在同一个@tla.kernel中声明独立的阶段同步flag
up_ready = tla.cross_flag("ffn_up_ready", mode=2)
all_aic_arrived = tla.cross_flag("ffn_all_aic_arrived", mode=0)
down_release = tla.cross_flag("ffn_down_release", mode=2)

with tla.cube():
    # 位于AIC上采样之后、下采样之前
    tla.pipe_barrier(tla.arch.ALL)
    tla.cross_core_wait_flag(up_ready, tla.arch.MTE2)
    tla.cross_core_set_flag(all_aic_arrived, tla.arch.MTE2)
    tla.cross_core_wait_flag(all_aic_arrived, tla.arch.MTE2)
    tla.cross_core_set_flag(down_release, tla.arch.MTE2)
    # 随后在MTE2读取下采样输入

with tla.vector():
    # 位于AIV的GELU及hidden写回之后、下采样阶段之前
    tla.pipe_barrier(tla.arch.ALL)
    tla.cross_core_set_flag(up_ready, tla.arch.MTE3)
    tla.cross_core_wait_flag(down_release, tla.arch.MTE2)
    tla.pipe_barrier(tla.arch.ALL)
    # 随后进入下采样阶段，Basic路径可以没有AIV计算
```

cube/vector区域分别编译到AIC/AIV执行，以上排列不表示先执行完全部cube代码才执行vector代码
`PIPE_ALL`只负责本核收尾，跨核关系由mode0/mode2的set/wait建立；跨核指令选择具体流水，不以PIPE_ALL作为其pipe参数
AIC报到、汇合、放行和下采样读入按MTE2顺序组织，避免其他流水越过阶段等待

CATLASS C++的`CrossCoreBarrier`提供mode0等封装，但固定版本没有mode2的`BarrierFlag`特化，不能直接假设`CrossCoreBarrier<2>`可调用
这里的mode2用显式set/wait实现AIV→AIC报到和AIC→AIV放行[^D4]

### 5.2 同步验证约束

- 全部launch的AIC及对应两个AIV均参与一次协议，即使没有tile或尾块有效行数为0，也不能跳过同步
- 三个阶段flag使用独立名称，与MatMul内部流水、Stream-K归约flag统一检查分配，不硬编码可能冲突的ID
- mode2在AIC侧应lower为覆盖两个AIV的操作，检查生成IR和设备代码中的双路set/wait[^D2]
- 不在有分支次数差异的tile循环中重复此全局协议，各参与核的set/wait次数必须一致
- V1先在默认npu:0的单流场景验证，不承诺多流超额占核并发，跨核屏障须满足整组核可调度条件，按运行时能力核验batchmode配置[^D3]
- 验证包含核间不均衡、空闲核、奇数M尾块、全部下采样候选及重复调用，确认不死锁、不读取旧hidden、flag计数不残留

当前为源码与协议设计核查，未执行NPU编译、正确性或死锁测试，不据此宣称同步实现已验证

## 6. 输入兼容范围：基线事实与V1待确认项

### 6.1 非连续Tensor

基线Python插件对X、W1、W2调用`.contiguous()`后再传入融合算子，ACLNN准备阶段也存在连续化和格式处理[^B1][^B2]
因此属于接口层接受非连续输入、计算主体使用连续数据，不等于MatMul设备组件原生支持任意stride
非连续输入可能引入额外设备拷贝任务

V1已确认沿用入口连续化，“一次Kernel”限定为FFN主体，连续化可能产生的辅助任务另计

### 6.2 多维输入

基线允许X为2～8维，不是无限rank；最后一维为K，其他维度相乘得到M[^B2][^B3]

```text
x：[B,S,K]
逻辑矩阵X：[B*S,K]
计算输出Y：[B*S,N]
接口输出y：[B,S,N]
```

连续化之后，Tiling通过`GetBs`计算前导维乘积，设备按二维矩阵访问同一块连续存储，无需额外执行一个展平Kernel
插件创建输出时复制输入shape并把最后一维替换为N，因此输出保持前导维[^B1][^B3]

V1已确认沿用2～8维输入，前导维展平计算后恢复输出shape

### 6.3 非对齐尺寸

基线Ascend950非量化GELU接口不统一要求M/K/H/N为16或32的整数倍，设备实现包含尾块调度、K尾段及按有效宽度写回逻辑[^B2][^B7]
现有测试明确覆盖M=97和三维输入，但不能据此声称所有非对齐组合、所有下采样路径均已验证[^B4]
全载、Stream-K仍有各自的容量和shape门槛，接口支持尾块不代表任意shape都可强制使用这些路径

V1已确认支持非对齐尺寸，搬运、计算和写回均需处理有效尾块，不能越界读写
手动选择的全载或Stream-K路径若不满足适用条件，应明确报错，具体边界case随测试集合补充

### 6.4 布局

基线GELU支持Linear和Canonical两种权重布局，Canonical为W1=`[K,H]`、W2=`[H,N]`，计算不转置权重
V1是否同时保留两种布局，以及是否采用显式layout参数避免方阵歧义，尚未确认

## 7. 精度验证

沿用基线测试的主要验收方式，V1去掉两个Bias项，只测试GELU[^B4]

```python
hidden_fp32 = torch.nn.functional.gelu(x.float() @ weight1.float().T)
golden = (hidden_fp32 @ weight2.float().T).to(x.dtype)
diff = (output.float() - golden.float()).abs()
mean_rel = (diff / (golden.float().abs() + 1e-3)).mean()
# 基线主要判据：mean_rel < 0.05
```

注意：golden是全FP32计算后转换输出，设备hidden存在中间舍入，这两者不是同一计算精度路径
可增加分阶段舍入参考定位误差，但不替代以上基线判据
测试还须检查shape、dtype和有限值，记录最大绝对误差、均方根误差及平均相对误差，不能仅凭平均误差掩盖NaN/Inf

case全集待补充，拟覆盖FP16/BF16、四种下采样候选、适用性拒绝、非连续输入、2～8维及非对齐尺寸
正确性验证通过后才进入对应case的性能统计

## 8. 性能采集与分析

性能入口同时支持单次手动配置和外层搜索调优，默认设备为`npu:0`
采集FFN主体的task duration及工具支持的流水利用率，并记录case、dtype、路径、Tiling、软件版本和设备信息
首次JIT编译、预热和正式计时分开，profiler采集与不带profiler的计时分开，记录实际预热及重复次数
若存在连续化等辅助任务，单列耗时，不混同FFN主体task duration

搜索调优对每个case执行以下流程：

1. 枚举配置搜索空间中的全部上、下采样组合
2. 检查适用性，编译并验证精度，记录不适用、编译失败或精度失败的原因
3. 对通过的组合使用相同输入、设备和预热/重复次数进行计时，不并发运行候选造成资源干扰
4. 默认以不带profiler的重复设备计时中位数作为排序指标，选出耗时最小的组合；profiler的task duration和流水指标单列用于分析
5. 将全部组合结果写入`performance.csv`，将每个case的最优组合写入`best_configs.json`，支持后续按该组合手动复现

“最优”限定为当前case、dtype、布局、设备、软件版本和配置搜索空间内的实测最优，不声称所有可能Tiling中的全局最优
报告记录搜索空间、计时口径和完成状态；搜索中断时只标记当前最佳，全部候选均不可用时不生成有效最优配置

分析重点：上采样、GELU、hidden的GM读写、跨核同步等待、下采样及Stream-K归约
单个融合task的总时长不能直接当作各阶段时长，阶段瓶颈需结合可获得的profiling证据判断

目前没有case全集、性能标杆或加速比目标，均保留待补充，不将未测优化写成性能收益

## 9. 待确认清单

1. 权重布局：是否同时支持Linear和Canonical，是否使用显式layout参数
2. 同步验证：mode0＋mode2方案已确定，待在固定DSL及Ascend950上验证lowering、参与核和重复调用
3. 执行环境：CANN和DSL已安装，待提供连接方式并记录实际版本
4. case全集、性能对照、指标可用性和验收目标
5. CATLASS复用入口：A/B全载DSL样例位置，以及Basic/Stream-K已有设备逻辑如何在固定版本内组合为一个FFN Kernel

## 10. 参考依据

以下`B`类路径相对于MindIE-SD仓库根目录，固定提交为`4cb292aec256a3bb3c0deea51c5cf55426b1f0e7`
以下`D`类路径相对于`3rdparty/catlass`，固定为上述`v2.0.0`提交

[^B1]: `csrc/plugin/eagle_ffn_linear.cpp`，`eagle_ffn_linear_mindie_sd_impl_npu`，62–71行检查维度并连续化输入，94–108行保留输出前导维并调用ACLNN
[^B2]: `csrc/ops/eagle_ffn/op_host/op_api/aclnn_eagle_ffn.cpp`，`CheckFmapWeightShape`，275–353行检查2～8维及布局；`GetFFNResultByL0Api`，1021–1025行准备连续输入，1060行处理输出ViewCopy
[^B3]: `csrc/ops/eagle_ffn/op_host/ffn_tiling.cpp`，`FFNTiling::GetBs`，385–405行计算前导维乘积并检查M上界，529行从最后一维取得K
[^B4]: `tests/ops/eagle_ffn_linear/test_eagle_ffn_linear.py`，`_ref`，55–63行定义FP32参考；`_run_and_check`，81–88行定义输出转换及平均相对误差判据；69行设置M=97，116–118行测试多维输入
[^B5]: `csrc/ops/eagle_ffn/op_kernel/eagle_ffn_apt.cpp`，`FfnArch35KernelImpl`，288–335行依次执行GELU/SiLU上采样、307行全核同步及下采样，340行是统一设备入口
[^B6]: `csrc/ops/eagle_ffn/op_kernel/3rd/mat_mul_v3/arch35/cmct/epilogue/fusion/fusion_regbase_act.h`中的GELU实现；`cmct/epilogue/block_epilogue_elementwise.h`的`DoFusionAndCast`和`Run`负责hidden转换与写回，路径前缀同前
[^B7]: `csrc/ops/eagle_ffn/op_kernel/3rd/mat_mul_v3/arch35/block_scheduler_aswt.h`负责M/N尾块；同目录`cmct/block/block_mmad_pingpong_without_que.h`处理K块，`cmct/epilogue/block_epilogue_elementwise.h`的119–176行按有效行宽写GM
[^D1]: `python/tla_dsl/README.md`的“兼容性”及“快速开始”章节
[^D2]: `python/tla_dsl/catlass/core_api.py`，4526–4687行定义`cross_flag`及跨核set/wait，4742行定义`pipe_barrier`；`python/tla_dsl/csrc/mlir/lib/Passes/TlaLowerFlagBarrierToHivmPass.cpp`的`CrossUseOpConversion`实现mode0及mode2降低，mode2在AIC侧处理base和base＋16两路，当前仅静态核查
[^D3]: 本地asc-devkit仓库的`docs/zh/guide/编程指南/高级编程/硬件实现/架构规格/NPU架构版本3510.md`，306–309行说明mode0/1/2/4语义；`docs/zh/api/SIMD-API/c_api/sync/asc_sync_inter_arrive.md`的“约束说明”列出具体流水、flag计数及batchmode约束，实际执行仍以已安装CANN版本为准
[^D4]: `include/catlass/arch/cross_core_sync.hpp`，`BarrierFlag`特化及`CrossCoreBarrier`的set＋wait实现
[^D5]: 固定v2.0.0的`python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed_ub2l1.py`，`basic_mixed_ub2l1`在同一个`@tla.kernel`中包含cube/vector区域及cross_flag同步；`basic_mmad/basic_matmul.py:36`和`basic_mmad_streamk/basic_mmad_streamk.py:55`定义各自独立Kernel入口，后两者路径同属end_to_end

基线仓库：[MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
DSL版本：[CATLASS v2.0.0](https://gitcode.com/cann/catlass/tags/v2.0.0)
本项目补充材料：`MindIE-SD EagleFFN算子实现分析.md`、`ref/triton_impl.py`，后者为多Kernel参考，不能直接作为V1单Kernel实现
