# MindIE-SD EagleFFN算子实现分析

基线：MindIE-SD提交`4cb292aec256a3bb3c0deea51c5cf55426b1f0e7`，本地分支`pr_588`

第一、第二部分还原此提交的Ascend950实现；第三、第四部分是本次独立编写的数学等价参考方案，不是从原仓库导出的实现，也不声称复刻其性能
完整工程见[eagle_ffn_reference](eagle_ffn_reference/README.md)，原仓库及既有分析文档不作修改
文中源码路径均相对MindIE-SD仓库根目录，文末引用列出固定提交的文件、函数和行号

## 完整计算逻辑

EagleFFN的整体计算是：

$$
Y=\operatorname{EagleFFN}(X)
$$

它完成“上投影→激活或门控→下投影”，通过`activation`选择以下三种路径之一，**不是将GELU、SiLU和SwiGLU依次串联**

### 三种可选计算路径

以下使用Linear权重布局，`+b`表示将Bias广播加到每一行

当`activation="gelu"`时：

$$
Y=\operatorname{GELU}(XW_1^\mathsf T+b_1)W_2^\mathsf T+b_2
$$

当`activation="silu"`时：

$$
Y=\operatorname{SiLU}(XW_1^\mathsf T+b_1)W_2^\mathsf T+b_2
$$

当`activation="swiglu"`时：

$$
Y=\operatorname{SwiGLU}(XW_1^\mathsf T+b_1)W_2^\mathsf T+b_2
$$

前两种路径的上投影输出宽度为H；SwiGLU的上投影输出宽度为2H，门控相乘后变为H，再进行下投影


GELU对上投影结果逐元素计算：

$$
\operatorname{GELU}(z)=\frac{z}{2}\left[1+\operatorname{erf}\left(\frac{z}{\sqrt2}\right)\right]
$$

其中erf是误差函数，这里采用GELU的erf定义；原设备代码使用erf的有理式近似，详见第二部分[^K6]

SiLU对上投影结果逐元素计算：

$$
\operatorname{SiLU}(z)=\frac{z}{1+e^{-z}}
$$

SwiGLU将上投影结果按列分成等宽的gate和up两半，记为G和U，只对gate做SiLU，再与up逐元素相乘：

$$
\operatorname{SwiGLU}([G,U])=\operatorname{SiLU}(G)\odot U
$$

因此，SwiGLU路径也可以写成：

$$
Y=\left[\operatorname{SiLU}(XW_g^\mathsf T+b_g)\odot(XW_u^\mathsf T+b_u)\right]W_2^\mathsf T+b_2
$$

这里weight1由gate权重和up权重按行拼接：前H行为$W_g$，后H行为$W_u$，bias1同样按gate、up顺序拼接
两半不能交换，也不是对全部2H列施加SiLU

### 输入维度、布局与无Bias情形

X表示展平后的二维输入，shape为`[M,K]`：M是展平后的行数，K是输入宽度，H是激活后的隐藏宽度，N是输出宽度
对于多维输入x，将最后一维保留为K、前导维合并为M，即`X = x.reshape(-1, K)`，计算后将Y恢复为`[*x.shape[:-1], N]`

| 约定 | weight1 | weight2 | 矩阵乘 |
| --- | --- | --- | --- |
| Linear，GELU/SiLU | `[H,K]` | `[N,H]` | `X @ weight1.T`、`A @ weight2.T` |
| Canonical，GELU/SiLU | `[K,H]` | `[H,N]` | `X @ weight1`、`A @ weight2` |
| Linear，SwiGLU | `[2H,K]` | `[N,H]` | gate/up按行拼接 |

表中的A表示激活或门控后的hidden，shape为`[M,H]`
bias1的长度等于上投影输出宽度，即H或2H，bias2的长度为N
Canonical只是同一权重的转置存储，不是另一种算法；融合SwiGLU不支持Canonical
无Bias时直接去掉完整公式中的b1、b2，融合接口要求两个Bias同时有或同时无[^H1][^H3][^T1]

### 实际计算的精度边界

以上公式描述数学逻辑，实际实现还包含中间舍入：

- 输入、权重、hidden和输出使用FP16/BF16，矩阵累加和激活中间量主要使用FP32
- 分阶段参考先用FP32完成上投影、Bias和激活或门控，再将hidden转换为输入dtype
- 下投影读取转换后的hidden，用FP32计算并加Bias，最终输出再转换为输入dtype

这些转换会影响数值结果；即使公式相同，原Cube归约顺序、Bias注入方式、激活近似和torch/Triton实现也不保证逐位一致

## 第一部分：Host侧与网络接入

### 1.1 网络位置

典型Pre-Norm Transformer Block示意：

$$
R=X_0+\operatorname{Attention}(\operatorname{Norm}_1(X_0)),\quad
V=\operatorname{Norm}_2(R),\quad X_1=R+\operatorname{EagleFFN}(V)
$$

EagleFFN覆盖`上投影＋Bias1→激活/门控乘法→下投影＋Bias2`
Attention、Norm和网络外部残差不在算子范围内；残差相加需要N等于网络宽度，但算子本身不要求N=K

Diffusion Transformer还可能存在条件调制与门控残差：

$$
V=(1+s(c))\odot\operatorname{Norm}(R)+t(c),\qquad
X_1=R+g(c)\odot\operatorname{EagleFFN}(V)
$$

这是典型结构示意，不是PR已接入具体DiT模型的证据；此提交提供接口、注册和测试，不能由算子名推出模型接入已完成
外部门控$g(c)$与SwiGLU内部gate不是同一个量

### 1.2 如何替换已有Linear

~~~python
from mindiesd.layers.eagle_ffn_linear import eagle_ffn_linear

v = norm(after_attention_residual)
f = eagle_ffn_linear(
    v, up.weight, down.weight, up.bias, down.bias,
    activation="gelu", fused=True,
)
y = after_attention_residual + f
~~~

`nn.Linear(K,H)`和`nn.Linear(H,N)`的权重不需要转置
若SwiGLU原来分为gate/up两个Linear，在模型准备阶段按gate、up顺序拼接weight和Bias为`[2H,K]`、`[2H]`，不要每次forward重复拼接
原网络若包含dropout、额外缩放等操作，不能不加判断地替换为此接口[^H1]

### 1.3 Python→插件→ACLNN→设备调用链

| 层次 | 入口及动作 | 边界 |
| --- | --- | --- |
| 公共Python接口 | `eagle_ffn_linear`校验激活、`get_npu_device()`判断平台 | `fused && A5`走融合，否则`_chain_fallback` |
| Python算子桥 | `_custom_ops.eagle_ffn_linear`→`torch.ops.mindiesd.eagle_ffn_linear` | 公共接口传`inner_precise=0` |
| PyTorch注册 | `register_ops.cpp`注册schema和NPU实现 | fake只推导输出shape |
| C++插件 | `eagle_ffn_linear`检查、连续化、判断布局、创建输出 | 一次`EXEC_NPU_CMD(aclnnEagleFfnV2,...)` |
| ACLNN准备 | `aclnnEagleFfnV2GetWorkspaceSize`→`GetFFNResultByL0Api` | 输入准备、执行器组装、workspace计算 |
| L0包装 | `l0op::EagleFfn` | 一处`ADD_TO_LAUNCHER_LIST_AICORE(EagleFfn,...)` |
| ACLNN执行 | `aclnnEagleFfnV2`执行executor | 使用调用方stream |
| Tiling与设备 | `RunFusionKernelTiling`→`FFNArch35Tiling`→全局`eagle_ffn` | 主Kernel内串联多个组件 |

插件调用将expert和量化参数传null，不存在Host分别调用上投影、激活、下投影三次ACLNN的结构[^H1][^H2][^H3][^H4][^H5][^T0][^K1]

辅助操作必须另外计数：

| 操作 | 源码确定的调用 | 需运行时确认 |
| --- | --- | --- |
| 插件`.contiguous()` | 对输入、权重、Bias规整 | 已连续时是否无拷贝，实际拷贝下发数 |
| `InputsContiguousAndTransFormat` | ACLNN输入规整与转格式 | executor是否消除转换 |
| `OutputransFormat` | L0结果恢复格式 | 是否零拷贝 |
| `l0op::ViewCopy` | 结果关联/复制到调用方输出 | 是否生成独立设备任务 |

“主FFN Kernel下发1次”不等于“整个Python调用在Profiler中只有1条设备任务”[^H3][^H4]

### 1.4 组件数、launch数、融合与同步

下表针对非空输入，GEMM规模记为`(M,归约K,输出N)`
“组件”指设备内矩阵计算入口；事件握手次数依赖tile/K循环，不包括在顶层屏障次数中

| 路径 | 数学GEMM/设备内组件 | 主FFN launch | 顶层`SyncAll<false>` | 额外同步 |
| --- | --- | --- | --- | --- |
| GELU/SiLU＋普通down | up`(M,K,H)`＋down`(M,H,N)`，2次 | 1 | 1 | tile级AIC/AIV握手 |
| GELU/SiLU＋A/B全载down | 同上，2次，仅down供数策略变化 | 1 | 1 | 同上 |
| GELU/SiLU＋Stream-K down | 同上，2次，down内部AIC分片＋AIV归约 | 1 | 1 | 另有AIV分支一次`SyncAll()` |
| SwiGLU单上投影 | up`(M,K,2H)`＋down`(M,H,N)`，2次 | 1 | 1 | 无Stream-K |
| SwiGLU双上投影 | gate`(M,K,H)`＋up`(M,K,H)`＋down`(M,H,N)`，3次 | 1 | 2 | 无Stream-K |

up融合Bias1、激活和hidden转换；SwiGLU双上投影在第二个组件中读取gate并融合SiLU×up
down融合Bias2和输出转换；SK中Bias只加到第0个K分片，归约后转换
所有路径hidden落GM，双上投影还将FP32 gate落GM；没有独立激活launch或归约launch[^K1][^K2][^K7][^K8]

### 1.5 Tiling数据结构与选择顺序

先看候选组件及组合：Host只下发1个主FFN Kernel，内部按Tiling执行以下路径
图中常驻指矩阵全载到各参与核的L1，普通、A常驻和B常驻分别列出；不按dtype和布局进一步细分[^H5][^T1][^K1]

~~~mermaid
flowchart TD
    hostCall["Host计算Tiling，下发1个FFN Kernel"]

    subgraph ffnKernel ["单个FFN Kernel内部"]
        subgraph upStage ["上投影：5种组件，均非全载"]
            activationChoice{"activation"}
            upGelu["U1：MatMul＋Bias＋GELU"]
            upSilu["U2：MatMul＋Bias＋SiLU"]
            swigluChoice{"swigluSingle"}
            upSwigluSingle["U3：单次MatMul＋Bias＋SwiGLU"]
            upGate["U4：gate投影＋Bias，写FP32 gate到GM"]
            gateSync["SyncAll：gate就绪"]
            upSwigluSplit["U5：up投影＋Bias＋SiLU(gate)×up"]

            activationChoice -->|"GELU"| upGelu
            activationChoice -->|"SiLU"| upSilu
            activationChoice -->|"SwiGLU"| swigluChoice
            swigluChoice -->|"1：单上投影"| upSwigluSingle
            swigluChoice -->|"0：双上投影"| upGate
            upGate --> gateSync --> upSwigluSplit
        end

        hiddenGm["hidden写GM，FP16/BF16"]
        hiddenSync["SyncAll：hidden就绪"]

        subgraph downStage ["下投影：4种候选，均计算hidden×W2ᵀ＋Bias"]
            downChoice{"按Host Tiling选择"}
            downBasic["D1：普通MatMul，非全载"]
            downAFull["D2：A全载MatMul，hidden常驻L1"]
            downBFull["D3：B全载MatMul，weight2常驻L1"]
            downStreamK["D4：Stream-K，K分片＋FP32局部和归约"]

            downChoice -->|"Basic，fullLoad=0"| downBasic
            downChoice -->|"Basic，fullLoad=1"| downAFull
            downChoice -->|"Basic，fullLoad=2"| downBFull
            downChoice -->|"Stream-K，仅GELU/SiLU"| downStreamK
        end

        upGelu --> hiddenGm
        upSilu --> hiddenGm
        upSwigluSingle --> hiddenGm
        upSwigluSplit --> hiddenGm
        hiddenGm --> hiddenSync --> downChoice
    end

    outputY["输出Y"]
    hostCall --> activationChoice
    downBasic --> outputY
    downAFull --> outputY
    downBFull --> outputY
    downStreamK --> outputY

    style ffnKernel fill:#F5F5F5,stroke:#888888
    style upStage fill:#EAF3FF,stroke:#6B9BD2
    style downStage fill:#EAF7EF,stroke:#6BA67C
~~~

每次选择一条上投影路径和一种下投影实现：通常执行2段组件，SwiGLU双上投影执行3段组件，不是将所有候选依次执行

| 类型 | 核心职责 |
| --- | --- |
| `MatMulV3Args` | m/k/n、dtype及字节数、转置、ND、Bias、策略偏好 |
| `MatmulV3CompileInfo` | AIC/AIV数、L1/L0A/L0C/UB/Bias Table容量和带宽模型参数 |
| `MatMulV3RunInfo` | 候选base块、step、depth、尾块、全载、缓冲和核数 |
| `FfnMatMulV3Tiling` | 派生Basic ASWT，`ComputeOnly`不提交独立MatMul launch，up禁止全载 |
| `FfnMatMulV3StreamKTiling` | 计算down的SK候选 |
| `MatMulV3BasicTilingData` | 每段设备消费的L1/L0块、尾块和模式 |
| `FFNTilingData` | up/down两套字段及hidden、布局、Bias、SwiGLU标志 |

FFN设置A不转置，Linear的B转置、Canonical的B不转置，ND，关闭HF32
偏好为`preferL0cDB2=false`、`preferL0cMSplitDB2=false`、`preferNoMSplit=false`、`preferUbDB2=true`
`swigluSingleNAlign32 = isSwiglu && n == n1`是数值比较，不能无条件说只针对up[^T1]

~~~text
检查A950、单expert、FP16/BF16、激活、布局、成对Bias
up = basic_tiling(M,K,n1, allow_full_load=false)
if SwiGLU:
    single = up.n==n1 && up.nL1%32==0 && up.baseN%32==0 && up.nL1==up.baseN
    if not single: up = basic_tiling(M,K,H, allow_full_load=false)
down = basic_tiling(M,H,N, allow_full_load=true)  # 内部A全载优先于B全载
if not SwiGLU and down.baseM < align16(M):
    sk = streamk_tiling(M,H,N)
    if sk成功 and sk.baseM >= align16(M): down=sk; mode=STREAMK
    elif sk不适用: 保留basic/full-load
    elif sk是其他错误: 返回失败
~~~

A950的`FFNArch35Tiling`失败即返回失败，不会转入文件内遗留的旧量化/MoE Tiling
Basic模式包含普通down和A/B全载down，不等于必然无全载[^T0][^T1]

### 1.6 Basic MatMul实际分块

本节用单段矩阵乘的m/k/n，down时k=H
记$a_q(v)=q\lceil v/q\rceil$、$f_q(v)=q\lfloor v/q\rfloor$，输入字节数s=2，可用AIC数C，容量$L_1,L_A,L_C,U_B,B_T$均以字节计
以下化简限定FP16/BF16、A不转置、ND，不代表通用MatMul所有分支

#### 1.6.1 基础块搜索

`ResetBase`默认baseM/N/K为128、256、128/s，DAV_3510将baseM改为256，但随后`GetRebalanceBlock`会重新搜索
候选满足：

$$
4b_Mb_N\le L_{\rm tile},\quad
2sb_Mb_K\le L_A,\quad 2sb_Nb_K\le L_A
$$

$L_{\rm tile}$通常为L0C，部分不足核场景取$\min(L_C,U_B)$
不是选最大块就结束，还使用带宽/均衡模型：

$$
c_{mn}=(m+n)/(mn),\quad \rho=\max((m+n)ks/L2Size,1),\quad P=f_{\rm cube}\cdot8C
$$

$$
E=BW_{L2}/P+\rho(1-BW_{L2}/BW_{HBM})c_{mn}-(1+BW_{L2}/BW_{HBM})/k
$$

初始$1/b_M+1/b_N>E$视为memory-bound，候选搜索前边界再乘`CUBE_BOUND_RATIO=0.85`
M按16对齐；令$k_{\rm edge}=mnBW_{HBM}/((m+n)BW_{L2})$，若$k<k_{\rm edge}$，N按256/s对齐，否则Linear按16、Canonical按innerAlign/s
memory-bound时innerAlign=128字节，否则64字节，SwiGLU配对标记使N对齐单位至少32

`GetMaxBaseWithLimit`还限制每个方向的上界，令：

$$
k_{\min}=\min(\text{memory-bound? }16:128/s,a_{16}(k))s
$$

$$
b_{\max}\le\min\left(\frac{L_A}{2k_{\min}},
\frac{L_{\rm tile}}{4\cdot16},
\frac{L_1}{4s\min(k_{\rm align},a_{16}(k))}\right)
$$

这里$k_{\rm align}$为memory-bound时256/s，否则512/s；N有Bias时还夹到$B_T/(2\cdot4)$，最后按实际shape和对齐约束裁剪[^T2]

#### 1.6.2 均衡率、K块和双缓冲

令$T=\lceil m/b_M\rceil\lceil n/b_N\rceil$，$q=\lceil T/C\rceil-1$，$t=T-qC$，$r_s=\lfloor C/t\rfloor$
没有可用尾轮细分时均衡率$B=(mn/C)/((q+1)b_Mb_N)$
有主轮且尾分片面积不小于4096时，令$r=\lfloor\sqrt{r_s}\rfloor$、$o=\lfloor(r_s-r^2)/r\rfloor+1$：

$$
B=\frac{mn/C}{(q+1/[r(r+o-1)])b_Mb_N}
$$

搜索以达到Cube-bound后的均衡改善，或更小的$(1/b_M+1/b_N)/B$为更新条件，另有90%均衡率剪枝和epsilon平局规则
要复刻同一tiling结果，需保持候选遍历和比较顺序，不仅是容量约束

$$
k_{\max}=\left\lfloor L_A/(2s\max(b_M,b_N))\right\rfloor
$$

若$a_{16}(k)\le k_{\max}$取$b_K=a_{16}(k)$，否则优先$f_{256/s}(k_{\max})$，不够该传输单位时从128、64、32、16中选可行值
`preferUbDB2`且$b_N>32,4b_Mb_N>U_B$时，先将N块减半向下对齐32一次，再计算K块

$$
usedCoreNum=\min(C,\lceil m/b_M\rceil\lceil n/b_N\rceil)
$$

$$
l0cDB=(8b_Mb_N\le L_C?2:1),\qquad ubDB=(4b_Mb_N\le U_B?2:1)
$$

UB判定是在两个AIV分担行的语境中，不能按“每个AIV各装完整两份结果”解释；带宽模型的缺省值也不是实测规格[^T2]

#### 1.6.3 L1块与尾块

默认L1搜索扣Bias预留$2b_N\cdot4$，扫描$j=1\ldots\min(\lceil k/b_K\rceil,8)$：

$$
q_K=jb_K,\ S_A=b_Mq_Ks,\ S_B=b_Nq_Ks,\quad
2(S_A+S_B)\le L_{\rm remain},\quad4\max(S_A,S_B)\le L_1
$$

优先256/512字节K传输对齐，并考虑单次搬运32768字节门槛，不是无条件取最大j
`stepKa=stepKb=j`，`depthA1=depthB1=2j`
Basic ASWT的非全载ON_THE_FLY分支随后覆盖为：

$$
j=\left\lfloor\frac{L_1-\mathbf1_{\rm bias}\cdot256\cdot4}
{2(b_M+b_N)b_Ks}\right\rfloor
$$

序列化时：

$$
mL1=\min(a_{16}(m),b_MstepM),\quad nL1=\min(a_{16}(n),b_NstepN),\quad
kL1=b_K\min(4,stepKa,stepKb)
$$

不能只看默认`CalL1Tiling`就认为得到了最终值；设备接收L1形状、`l1BufferNum`和L0C/UB标志，而不是原样接收step/depth[^T1][^T2][^T3][^T6]

边界有效块长度为$\min(b_M,m-m_0)$、$\min(b_N,n-n_0)$、$\min(b_K,k-k_0)$
ASWT另将最后一轮tile分给更多核，在$T>C,T\bmod C\ne0$时，满足
$(mTailCnt+1)nTailCnt(T\bmod C)\le C$及搬运限制才增加M切分，N类似
SwiGLU配对标记禁止此处N尾轮细分
`mTailCnt/nTailCnt`与`mBaseTailSplitCnt/nBaseTailSplitCnt/mTailMain/nTailMain`是两组不同字段，前者为尾轮分核，后者用于描述尾基础块重分布
本次选用的Basic链中后者由ResetBase设为1/1/0/0，全载也重置这些值，不应据字段存在推断启用了另一种尾基础块重分布优化[^T2][^T3][^T4][^T6][^K3]

### 1.7 A/B全载重算

只有down允许全载，检查A全载优先于B全载，都受ON_THE_FLY和供数代价条件约束

| 项 | A全载 | B全载 |
| --- | --- | --- |
| 常驻对象 | down的hidden | down的weight2 |
| 容量条件 | $a_{16}(m)a_{16}(k)s+S_{bias,A}\le3L_1/4$ | $a_{16}(n)a_{16}(k)s+S_{bias,B}\le3L_1/4$ |
| 分核条件 | `ceil(n/singleCoreN)>C`，排除`k<=128 && mCnt!=1` | `ceil(m/singleCoreM)>C` |
| Bias预留 | $2b_Nsizeof(BiasT)$ | $a_{16}(n)sizeof(BiasT)$ |
| 全载侧step | `stepM=ceil(align16(m)/bM)`、`stepKa=ceil(align16(k)/bK)` | `stepN=ceil(align16(n)/bN)`、`stepKb=ceil(align16(k)/bK)` |
| 单核形状 | `singleCoreM=m,singleCoreN=bN` | `singleCoreM=bM,singleCoreN=n` |
| 核数 | `min(ceil(n/bN),C)` | `min(ceil(m/bM),C)` |
| 尾轮均衡方向 | N | M |

必须重算流式一侧的块，以A全载为例：

$$
L_{\rm free}=L_1-S_{\rm fullA}-S_{bias,A}
$$

$$
b_N'=\min\left(b_N,\ f_{16}\left(\min\left(
\frac{L_{\rm free}}{2b_Ks},\frac{L_C}{8b_M}\right)\right),
a_{16}(\lceil n/C\rceil)\right)
$$

Canonical另有128/s对齐收缩，B全载相应重算M
非全载侧stepK最大受$\min(\lceil k/b_K\rceil,L_{\rm free}/(2b_{\rm other}b_Ks),4)$限制
还有特定转置下baseK翻倍、达到32768字节搬运门槛后停止的规则
4份流式L1块＋常驻矩阵＋Bias放得下则`l1BufferNum=4`，否则为2，按新块重算L0C/UB缓冲
全载是每个参与核的L1驻留，不是全芯片共享一次加载，不改变FFN阶段数[^T3]

### 1.8 Stream-K条件与分块

此处归约k=H，通用能力要求ND、非self-noncontiguous、deterministic配置不大于1、AIV数为AIC的2倍
令$T_{256}=\lceil m/256\rceil\lceil n/256\rceil$：

- 纯SK：$T_{256}\le C/2$，且$a_{256}(k)\ge\max(8192,256C)/s$
- DP＋SK：m/n均为256倍数，$T_{256}\ge C$，$0<T_{256}\bmod C\le C/2$，且$k\ge\max(8192,128C)/s$

纯SK以256初始化M/N块，若某方向块数在$(C/3,C/2)$内提高到C/2，再计算：

$$
b_M=a_{16}(\lceil m/mCnt\rceil),\quad b_N=a_{16}(\lceil n/nCnt\rceil),\quad
q_K=\lfloor C/(mCnt\,nCnt)\rfloor,\quad skSingleCoreK=\lceil k/q_K\rceil
$$

DP＋SK对尾轮块数$t=T_{256}\bmod C$取$q_K=\lfloor C/t\rfloor$，算$skSingleCoreK=\lceil k/q_K\rceil$，再重算实际分片数$\lceil k/skSingleCoreK\rceil$

$$
b_K=\min\left(skSingleCoreK,\ f_{128/s}
\left(\frac{65536}{2s\max(b_M,b_N)}\right)\right)
$$

65536是源码`L0A_SIZE_2`常量；再算L1 tiling，DP＋SK有Bias时stepKa/Kb设3
FFN额外要求Basic的baseM小于$a_{16}(M)$而SK的baseM能够覆盖它；多M块的底层DP＋SK能力不能直接当成EagleFFN可达能力
SwiGLU不尝试SK[^T1][^T5]

### 1.9 TilingKey、blockDim与workspace

$$
blockDim=\max(up.usedCoreNum,down.usedCoreNum)
$$

Key用`GET_TPL_TILING_KEY(dtype,act,mode)`生成，共10种组合：2种dtype×GELU/SiLU×Basic/SK，加2种dtype×SwiGLU×Basic
Bias、转置、全载和SwiGLU single继续由字段选择，不各自增加Key维度[^T1][^K9]

设系统workspace为U字节，用户起点$P=GetUserWorkspace(workSpace)$，$D=2MH$：

$$
O=hiddenOffset=
\begin{cases}a_{128}(4MH)&\text{GELU/SiLU或SwiGLU双上投影}\\
0&\text{SwiGLU单上投影}\end{cases}
$$

| 区域 | shape/dtype | 字节数 | 相对P偏移 | 生命周期/实际用途 |
| --- | --- | --- | --- | --- |
| 系统区域 | 不解释为用户Tensor | U | P之前由框架管理 | 主Kernel |
| 双上投影gate | `[M,H]`/FP32 | 有效4MH，预留O | 0 | gate写，第二up读取 |
| GELU/SiLU头部预留 | 可容纳`[M,H]`/FP32 | O | 0 | 当前mixed up走L0C→UB，不将完整Z写入此区 |
| single头部 | 无 | 0 | 0 | 无gate GM缓冲 |
| hidden | `[M,H]`/输入dtype | D | O | 激活写完，down读 |
| 固定间隔 | 非Tensor | 128 | O+D | 保留区，不等于对齐操作 |
| SK scratch | 每核至多`[256,256]`/FP32 | `down.usedCoreNum*256*256*4` | O+D+128 | 局部和写，AIV归约读 |
| 非SK尾部预留 | 可容纳`[M,N]`/FP32 | 4MN | O+D+128 | 当前普通/full-load down传空workspace，不据此认定FP32 Y落GM |

$$
workspaceBytes=U+O+D+128+
\begin{cases}down.usedCoreNum\cdot256^2\cdot4&SK\\4MN&非SK\end{cases}
$$

O明确按128对齐，但D未对齐，因此`O+D+128`不保证128字节对齐，任意奇数MH甚至不保证4字节对齐
SK可达shape另受门槛限制，移植时仍须验证地址约束，不能概括为“所有区都按128对齐”
原hiddenOffset计算包含uint32乘法，超大shape还应检查溢出；workspace预算不等于实际GM流量，不包含输入、权重和最终输出[^T1][^K1][^K2]

### 1.10 Host计算→Tiling字段→Kernel消费

| Host来源 | 字段，逐段字段带up/down前缀 | 设备消费 |
| --- | --- | --- |
| 展平/布局 | `M/N/K,transB` | `MakeFfnMMTiling`及LayoutB选择 |
| 候选核数 | `UsedCoreNum` | 总launch取最大值；SK直接用down核数，普通/全载调度实际读`GetBlockNum()` |
| 基础块搜索 | `BaseM/BaseN/BaseK` | L0块与K循环 |
| step→L1形状 | `ML1/NL1/KL1` | L1供数和tile数 |
| 尾块均衡 | `MTailCnt/NTailCnt/MBaseTailSplitCnt/NBaseTailSplitCnt/MTailMain/NTailMain` | ASWT有效shape及offset |
| 缓冲容量 | `L1BufferNum/L0cDB/UbDB` | 前两项控制L1/L0C；当前mixed epilogue没有读取`GetUbDB()`来切换双槽 |
| 全载选择 | `FullLoad` | down模板fullLoad=0/1/2 |
| SK切K | `downSkSingleCoreK` | K起点和最后分片长度 |
| Bias/dtype | `hasBias/isFp16/biasIsFp16/biasIsBf16` | BiasT和Bias指针选择 |
| SwiGLU配对 | `swigluSingle` | 单2H或gate/up两段 |
| workspace | `hiddenOffset/hiddenRows/hiddenCols` | gate、hidden、downWs地址 |

`MakeFfnMMTiling`不是无损复制通用MatMul数据：`sliceM/srcNdStride`置0、`innerBatch`置1、L2模式置默认，读通用代码必须与FFN桥接交叉核对[^T1][^T6][^K1]

特别注意：普通up/down使用总launch的`GetBlockNum()`构造ASWT调度器，并用`min(tileNum,blockNum)`排除无任务核，不是各自强制按`up/downUsedCoreNum`发起一次子launch
`ubDB`虽然被序列化并存入调度器，当前mixed组件没有用它选择UB双槽，不能由Host偏好推导出实际跨tile双帧流水[^K2][^K3][^K10]

## 第二部分：Kernel侧现有实现


### 2.0 概括(以Activation Gelu为例)

1. 基线：GM->(Matmul-Up+Gelu) -> GM-> Matmul-Down
	1. Matmul-Up+Gelu: BasicMatmul+Gelu
	2. Matmul-Down: BasicMatmul or FullLoad or StreamK 
2. 开发计划：
	1. 实现一个dsl版本，逻辑和基线相同
	2. 尝试其他不同的matmul kernel组合
	3. 改进融合逻辑，对于小case做完gelu，直接从ub到L1









### 2.1 设备入口与组件组合

全局`eagle_ffn<DTYPE,ACT,MODE>`执行`InitSocState`，声明`KERNEL_TYPE_MIX_AIC_1_2`，读取Tiling后选择输入T和BiasT
BiasT由`biasIsFp16/biasIsBf16`决定，否则为float；无Bias时传空指针
`FfnArch35KernelImpl`通过`MakeFfnMMTiling`恢复up/down结构，按布局、模式和SwiGLU标记调用组件[^K1]

| 组件层 | 代表性类型 | 数据结构承担的逻辑 |
| --- | --- | --- |
| 混合上投影 | `KernelMatmulMixWithoutQue` | `Params`组合problemShape、MMAD参数、epilogue参数和scheduler参数 |
| 普通/全载down | `KernelMatmulWithoutQue` | AIC-only计算，`BlockEpilogueEmpty`，输出直接写GM |
| SK down | `KernelMatmulStreamK` | AIC产出部分和，AIV读scratch归约 |
| 调度 | `BuiltInAswtScheduler<FULL_LOAD_MODE>`→`BlockSchedulerAswtBuiltIn` | 保存L1/L0形状、尾块、窗口、块坐标 |
| SK调度 | `BuiltInStreamKScheduler`→`BlockSchedulerStreamKBuiltIn` | 二维tile与K分片共同组成任务 |
| MMAD配置 | `BlockMmadBuilder`＋`MatmulMultiBlockWithOutQue` | 类型、Layout、全载/融合策略组成`BlockMmad`特化 |
| 激活后处理 | `BlockEpilogueElementwise<...,FusionOp>` | UB视图、有效块、转换和输出 |

这些都是全局Kernel内实例化的C++对象与inline调用
其中名为`MatMulActKernel`的函数也不是另一次设备launch[^K1][^K2][^K3][^K4][^K10]

### 2.2 Basic块级计算与调度

对任意一段$C=AB+b$，$A\in\mathbb R^{m\times k},B\in\mathbb R^{k\times n}$，二维tile $I,J$的结果：

$$
C_{I,J}=\sum_{q=0}^{\lceil k/b_K\rceil-1}
A_{I,[qb_K:(q+1)b_K)}B_{[qb_K:(q+1)b_K),J}+\mathbf1b_J^\mathsf T
$$

尾块截断到真实k，Bias只加入一次，不能每轮K都加Bias
上投影再计算$R_T(\phi(C_{I,J}))$，普通down将$R_T(C_{I,J})$直接写Y

`BlockSchedulerAswtBuiltIn`存储`mTileNum/nTileNum/kTileNum`、L1/L0块、尾基础块、末轮切分和窗口参数
外层`tileIdx=coreId; tileIdx<tileNum; tileIdx+=GetBlockNum()`，同一核循环处理多个tile
二维tile不是简单行优先，而是M方向最多4行的窗口扫描，窗口内遍历N，奇数窗口反向扫描N
其非尾窗口主公式为：

$$
w=\min(4,mTileNum),\quad row=\lfloor tileIdx/(nTileNum\,w)\rfloor
$$

$$
mIdx=row\cdot w+(tileIdx\bmod w),\quad
nIdx=\lfloor tileIdx/w\rfloor\bmod nTileNum
$$

奇数row令$nIdx=nTileNum-1-nIdx$，最后窗口使用实际tailWindow替换w
末轮细分先重映射tileIdx，再叠加M/N切分offset；这是一种任务扫描重排，不是存储Layout的Swizzle[^K3]

伪代码：

~~~text
for tile in ASWT.tasks(coreId, launchBlockDim):
    I,J = scheduler.coordinate_and_valid_shape(tile)
    for L0_m, L0_n inside L1 tile:
        accumulator = 0  # 有Bias时由首次MMAD的Bias路径初始化
        for k1 in chunks(k, kL1):
            GM -> L1: A片段、B片段，首轮准备Bias
            for k0 in chunks(k1, baseK):
                L1 -> L0A/L0B
                accumulator += MMAD(A0,B0)  # L0C，FP32
        if fused_up:
            L0C -> paired AIV UB: FP32
            activation + cast -> GM hidden
        else:
            Fixpipe: L0C -> GM Y，转换为T
~~~

数据供给为GM→L1→L0A/L0B→MMAD/L0C
`CopyInA1/CopyInB1`处理ND到片上组织，`CopyInA2/CopyInB2`提供L0块；实际offset随Linear/Canonical变化，不等于Host必须先转置整张权重[^K4][^K10]

### 2.3 L1/L0缓冲与硬事件

`block_mmad_pingpong_without_que.h`是基本搬运/计算实现，保存`l1Local/l0aLocal/l0bLocal/l0cLocal`、L1循环计数、L0 ping-pong计数和Bias槽
普通2-buffer与4-buffer的L1布局不同，不能只把buffer数改大而保留旧偏移
本路径非全载的`l1BufferNum`保持`MatMulV3RunInfo`默认值2，4-buffer选择在全载重算中显式设置[^T2][^T3]

- L1槽号为`abL1LoopCnt & (l1BufNum-1)`，只用于2或4这样的幂次槽数
- L0A/L0B用`l0PingPong & 1`交替，MMAD消费结束后才允许重新装填
- `MTE1_MTE2`：L1数据已被搬到L0，允许GM→L1覆盖
- `MTE2_MTE1`：GM→L1完成，允许L1→L0读取
- `M_MTE1`：MMAD已消费L0A/B，允许MTE1覆盖
- `MTE1_M`：L0A/B准备好，允许MMAD
- `FIX_M`：Fixpipe已消费L0C，允许再次写对应L0C槽

缓冲容量只代表可驻留/复用槽位，不直接等于多路矩阵指令同时执行；事件保护的是生产者/消费者依赖[^K4]

### 2.4 AIC/AIV协作及真正融合位置

GELU上投影组装`MatmulMultiBlockWithOutQue<...,0,OP_TYPE_GELU_ERF>`，MMAD输出类型明确为float
`KernelMatmulMixWithoutQue::RunMmad`把epilogue的LocalTensor传给MMAD；结果由Fixpipe写到对应AIV的UB，而不是先写完整Z到GM再启动激活
一个AIC配两个AIV，AIV用`GetBlockIdx()/GetTaskRation()`确定对应Cube组，每个AIV处理约一半M行[^K2][^K5]

每个输出块：

1. AIC等待上一块AIV完成，避免覆盖相同UB
2. AIC计算包含Bias的FP32矩阵块，经Fixpipe写UB，向两个AIV发送CrossCore flag
3. AIV等待对应flag，执行激活/乘法/转换，并将hidden写GM
4. AIV通知槽位可复用，AIC继续；组件结束时排空在途标志
5. up阶段返回后，所有参与者执行顶层`SyncAll<false>()`，确保down读到完整hidden

flag轮换还依赖计数和ID分组，不是给每个tile无限分配新flag
此版本显式保留“每次MMAD前等待前一次epilogue”的安全顺序，不能凭`ubDB=2`删等待并宣称跨tile双帧重叠[^K2]

Bias通过GM→L1→Bias Table进入首次MMAD，`NeedBias(iter0,iter1)`保护只注入一次
`BlockEpilogueElementwise::DoFusionAndCast`在RegBase路径中将激活和CAST_RINT合并到VF寄存器计算并就地写回窄类型视图；非RegBase路径先算FP32再转换[^K4][^K5]

### 2.5 原激活的有限精度表达

GELU RegBase使用$t=\operatorname{clip}(z/\sqrt2,-3.92,3.92)$、$u=t^2$：

$$
\widetilde{\operatorname{erf}}(t)=
t\frac{P_0+P_1u+P_2u^2+P_3u^3+P_4u^4+P_5u^5}
{Q_0+Q_1u+Q_2u^2+Q_3u^3+Q_4u^4+u^5}
$$

| 系数 | 值 |
| --- | --- |
| P0,P1,P2 | 29639.384698、5063.7915060、1393.8061484 |
| P3,P4,P5 | 101.62808918、7.5517016694、0.053443748819 |
| Q0,Q1,Q2 | 26267.224157、13243.365831、3023.1248150 |
| Q3,Q4 | 398.56963806、31.212858877 |

最后$a=R_T(0.5z[1+\widetilde{\operatorname{erf}}(t)])$，外部乘数用原始z，不是clip后的值
SiLU使用FP32指数/加法/除法链，再CAST_RINT；SwiGLU single在同一VF中计算gate的SiLU并乘up，然后转换
这些实现的指令级舍入和库erf/sigmoid不必相同[^K6]

### 2.6 SwiGLU两种组织

单上投影计算的逻辑输出宽2H，但B加载会把每个tile对应的gate/up半块一起安排到L1局部布局，而不是先算完所有gate列才计算up列
Bias也按两半组织，epilogue消费对应gate/up对，输出宽度减半
尾块保留真实列数并补齐配对约束；这解释了Host要求baseN/nL1为32倍数且nL1=baseN

$$
A_{I,J}=R_T\left(
\operatorname{SiLU}\left(\sum_kX_{I,k}W_{g,J,k}+b_{g,J}\right)
\odot\left(\sum_kX_{I,k}W_{u,J,k}+b_{u,J}\right)\right)
$$

若单上投影条件不满足：

~~~text
RunFfnRawGateUp: MMAD(X,Wg)+bg -> FP32 gate[M,H] in GM
SyncAll<false>()
RunFfnSwigluUp: MMAD(X,Wu)+bu -> FP32 UB
               GM gate -> UB -> SiLU(gate)*up -> cast T -> GM hidden
SyncAll<false>()
AIC RunFfnDownMMFullLoad: hidden*W2^T+b2 -> Y
~~~

第一段使用`FusionCopy<float,float>`的实际实例化，gate并未提前转换成FP16/BF16；不能被文件里通用“fp32→bf16”注释误导
第二段`FusionSwiglu`使用`DataCopyPad`读FP32 gate，再指数、倒数和乘法
双上投影增加一次完整gate写/读及一次顶层屏障，但仍然只有一个FFN launch[^K1][^K4][^K5][^K6]

### 2.7 A/B全载的设备侧变化

`RunFfnDownMMFullLoad`按`down.fullLoad`选择`FULL_LOAD_MODE=1/2/0`，都进入`MatMulActKernel`→`KernelMatmulWithoutQue`

~~~text
A-full:
    每个参与AIC在tile循环前将整个当前A矩阵装入本地L1
    按N块循环；流式搬B，A从已驻留L1切片送L0
B-full:
    每个参与AIC预载整个当前B及Bias到本地L1
    按M块循环；流式搬A，B从已驻留L1切片送L0
两者:
    L0仍按baseK循环做FP32累加
    Bias只加一次，Fixpipe转换并输出Y
~~~

计算公式与Basic完全相同，只改变数据复用方向、流式块、buffer布局和任务粒度
这里`BlockEpilogueEmpty`并不表示“没有Bias/转换”，而是这些操作已经由MMAD及其输出通路完成，无独立AIV激活后处理[^K4][^K10]

### 2.8 Stream-K局部和、scratch与归约

令$s_K=down.skSingleCoreK$，分片r的归约范围
$K_r=[rs_K,\min((r+1)s_K,H))$
对SK输出块：

$$
P^{(r)}_{I,J}=\sum_{h\in K_r}\widehat A_{I,h}(W_2)_{J,h}
+\mathbf1_{r=0}\mathbf1(b_2)_J^\mathsf T,\qquad
Y_{I,J}=R_T\left(\sum_rP^{(r)}_{I,J}\right)
$$

完整二维整轮DP块可直接输出Y，只有尾轮SK块需要FP32 scratch及归约
`BlockSchedulerStreamKBuiltIn`计算$T=mTileNum\,nTileNum$、$t=T\bmod usedCoreNum$、$q=\lceil H/s_K\rceil$，FFN的batch=1时：

$$
T_{\rm DP}=T-t,\qquad T_{\rm tasks}=T_{\rm DP}+tq
$$

调度器为尾轮任务给出M/N/K坐标，二维坐标同样采用4行窗口；组件中还有DP＋SK预加载的任务交换，不改变结果所属坐标
FP32槽位以256×256元素为固定步长，对尾轮二维块j和K分片r，逻辑偏移为$(jq+r)256^2$，不能按有效尾块面积紧密拼接[^K7]

~~~text
AIC:
    for owned task (output tile, K slice):
        accumulate FP32 partial, add bias only if slice_id==0
        if DP: convert T and write Y
        else: write FP32 scratch slot
    signal paired AIV
AIV:
    wait producer signal
    SyncAll()  # 也涵盖没有归约任务的AIV，避免漏同步
    for assigned valid output elements:
        load all K-slice partials from GM to UB
        add in FP32
        CAST_RINT to T
        write Y
~~~

不是atomicAdd归约，也没有第三次归约launch
`block_mmad_streamk.h`中`BlockMmad`特化的`NeedBias`要求首个L1/L0 K迭代及`kCntIndex==0`，`BlockEpilogueStreamK`不再加Bias
归约时按有效行列处理，固定scratch槽中的padding不应作为真实输出参与计算[^K7][^K8]

### 2.9 支持限制与实现边界

- 当前算子注册面向ascend950，输入/权重/输出FP16或BF16，Bias为同类型或FP32；旧代码出现INT8、expert、量化参数不代表此A950融合路径支持它们
- SwiGLU仅Linear，gate/up按前后半行约定，不接受Canonical右半列切片
- Python fallback使用`F.linear`，不适配Canonical；与融合接口自动布局识别存在差异
- 原自动布局判断对方阵参考weight2消歧，仍可能无法表达调用者意图；新参考接口用显式layout
- 原mixed上投影移除了small-N的GM中转备选路径，文件说明以N≥16为目标；不应由通用接口推断所有极小hidden均已设备验证
- 底层`CanImplement`、遗留分支或存在字段不等于FFN一定调用/使用它，尤其是UB双缓冲和每段核数
- 本文是固定提交静态还原，没有提供原算子在Ascend950上的Profiler或性能实测[^H1][^H3][^T0][^T1][^K2][^K5]

## 第三部分：PyTorch小算子参考实现

### 3.1 模块和公共接口

工程是独立源码，不调用现有EagleFFN，也不依赖MindIE-SD安装

| 工程文件 | 职责 |
| --- | --- |
| `eagle_ffn/common.py` | `prepare`检查、展平、显式布局规范化 |
| `eagle_ffn/torch_impl.py` | `ffn_torch`、`ffn_torch_fp32`及小算子计算 |
| `eagle_ffn/triton_impl.py` | 两个JIT计算Kernel及下发包装 |
| `eagle_ffn/metrics.py` | 误差指标 |
| `example.py` | Norm和残差外置的网络子层示例 |
| `tests/test_reference.py` | 组合正确性与非法输入，按设备情况执行NPU测试 |
| `verify.py` | 六组精度JSON报告及可选预热计时 |

~~~python
ffn_torch(x, weight1, weight2, bias1=None, bias2=None,
          activation="gelu", layout="linear")
ffn_torch_fp32(x, weight1, weight2, bias1=None, bias2=None,
               activation="gelu", layout="linear")
~~~

两个函数参数一致，前者输出输入dtype，后者输出FP32
全FP32参考的输入仍由传入的FP16/BF16扩展，不能恢复输入量化前的值
检查三种激活、两种输入dtype、同dtype权重、成对Bias及其dtype/shape/device、两种整体布局
支持多维/非连续输入及M=0；K/H/N要求正数，不覆盖旧ACLNN所有退化输入和量化接口
规范化将两层权重统一为连续Linear布局，Canonical输入的转置/连续化可能产生额外拷贝

### 3.2 小算子与数学对应

~~~text
prepare -> X.float(), W1.float()
Z = matmul(X, W1.T)
Z = Z + bias1.float()                        # 若存在
GELU: A = 0.5*Z*(1+erf(Z/sqrt(2)))
SiLU: A = Z*sigmoid(Z)
SwiGLU: G,U = chunk(Z,2); A = G*sigmoid(G)*U
staged only: A = A.to(input_dtype).float()    # hidden舍入
Y = matmul(A, W2.float().T) + bias2.float()
staged only: Y = Y.to(input_dtype)           # 输出舍入
reshape Y to original leading dimensions + N
~~~

这是用矩阵乘、广播加法、erf/sigmoid、切分和逐元素乘法拼接的参考，不直接调用`F.linear`或融合FFN
测试另使用`F.linear/F.gelu/F.silu`构造独立表达的oracle，避免测试只重复调用被测函数
普通低精度`nn.Linear→activation→nn.Linear`可能在激活前额外舍入Z，因此不将它与分阶段参考当作逐位等价实现
最小网络替换见`example.py::FFNSublayer.forward`，只有FFN主体换成参考接口，Norm和残差保持外部

## 第四部分：Triton-Ascend数学等价参考

### 4.1 目标、依赖与边界

目标是Ascend950上的Triton-Ascend，接口与`ffn_torch`相同，计算主体只使用Triton，不回调torch矩阵乘或原EagleFFN
本机没有NPU，此版本完成源码和静态检查，未完成Ascend950编译/运行验收
目标环境需按[Triton-Ascend官方仓库](https://github.com/triton-lang/triton-ascend)安装匹配的驱动、CANN、torch_npu和编译器，不能用普通CUDA Triton包替代

`prepare`与PyTorch共享，torch仅用于张量检查、视图/连续化和分配
Triton是推理实现，带梯度参数必须在`torch.no_grad()`中调用，没有自定义backward

### 4.2 Kernel1：上投影、Bias、激活

使用`BM=32,BN=64,BK=32`，H方向输出块宽为BN，不将SwiGLU的2H直接作为hidden宽度
单program计算：

$$
Z_g\in\mathbb R^{BM\times BN}
=\sum_{q=0}^{\lceil K/BK\rceil-1}X_{I,Q_q}(W_g)_{J,Q_q}^\mathsf T+b_{g,J}
$$

GELU/SiLU将这里的Wg视为W1，然后激活
SwiGLU在相同program和相同K循环中再累加$Z_u$，读取weight1的H偏移行，最后：

$$
hidden_{I,J}=R_T\bigl(Z_g\sigma(Z_g)\odot Z_u\bigr)
$$

输入和权重16位，`tl.dot`累加变量为FP32，Bias读取后扩展FP32，激活后显式转换到hidden指针元素类型
行、列和归约边界分别mask，越界载入0，store只写真实M/H范围
GELU使用[tl.erf](https://triton-lang.org/main/python-api/generated/triton.language.erf.html)表达数学函数，不照搬原Padé实现；其Ascend后端lowering仍需目标版本验证

### 4.3 Kernel2：下投影、Bias、输出

$$
Y_{I,J}=R_T\left(
\sum_{q=0}^{\lceil H/BK\rceil-1}hidden_{I,Q_q}(W_2)_{J,Q_q}^\mathsf T+b_{2,J}
\right)
$$

同样使用32×64×32固定块，FP32累加，归约结束后加Bias并转换
每个输出tile由一个program独占，不需要atomic、split-K scratch或归约Kernel
masked store保证N非64倍数、M非32倍数的尾块不越界

### 4.4 grid、任务循环与同步

令$C_{\rm core}$为后端报告的`num_aicore`：

$$
T_1=\lceil M/32\rceil\lceil H/64\rceil,\quad grid_1=(\min(C_{\rm core},T_1),)
$$

$$
T_2=\lceil M/32\rceil\lceil N/64\rceil,\quad grid_2=(\min(C_{\rm core},T_2),)
$$

program p循环领取`tile=p,p+grid_size,...`，二维坐标用行优先商余数
这是core数有界的任务循环，依据[官方编程指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/programming_guide.md)的`num_aicore`查询和核内分批组织方式；不把CUDA warp数映射成Ascend物理核心数

两次launch在同一当前NPU stream上按顺序下发，第二个Kernel依赖第一个完成的hidden，包装中不调用Host synchronize
hidden为GM中的`[M,H]`输入dtype张量，字节数2MH，不需要原算子的系统/门控/SK手工workspace
跨Kernel的顺序依赖取代原单Kernel内`SyncAll`，不代表两个Kernel同时运行
辅助连续化不算这两个计算Kernel，M=0时不下发计算Kernel

### 4.5 与现有实现的差异

| 项目 | 当前C++融合实现 | 新Triton参考 |
| --- | --- | --- |
| 非空主计算launch | 1 | 2 |
| 顶层阶段依赖 | 单Kernel内全局屏障 | 同一stream的顺序launch |
| hidden | GM，16位 | GM，16位 |
| SwiGLU | 单2H或gate/up双组件 | 每个program同时算gate/up，不写完整gate GM |
| 下投影 | Basic、A/B全载、受限SK | 普通分块GEMM |
| Tiling | 硬件容量/代价/尾块/全载/SK搜索 | 固定32×64×32＋mask |
| 片上流水 | 手写L1/L0/Fixpipe、AIC/AIV事件 | 由Triton-Ascend编译器lowering |
| Bias | 原MMAD首次迭代注入；SK只在第0片加入 | FP32归约结束后显式相加 |
| 激活精度 | 原VF/Padé/指令序列 | tl.erf/tl.sigmoid，数学等价非逐位复刻 |

本工程没有宣称固定块在Ascend950上最优，也没有宣称两launch比原一launch更快
块大小、资源占用、编译可行性和性能必须在目标软件栈确认

## 验证与复现

执行命令、环境和实测数字见[eagle_ffn_reference/VALIDATION.md](eagle_ffn_reference/VALIDATION.md)
CPU测试覆盖120种组合及非连续/方阵/非法输入；NPU缺失时明确skip，存在NPU但导入或编译失败则失败，不静默回退
误差同时报告：

$$
E_{\max}=\max_i|y_i-r_i|,\quad
E_{\rm RMS}=\sqrt{\tfrac1L\sum_i(y_i-r_i)^2},\quad
E_{\rm rel2}=\frac{\|y-r\|_2}{\max(\|r\|_2,10^{-8})}
$$

还报告逐点最大相对误差$\max_i|y_i-r_i|/\max(|r_i|,10^{-8})$，它在参考值接近0时可很大，不能单独判断整体精度
`verify.py --device npu:0 --triton --benchmark`先编译、检查正确性和预热，再在两端synchronize后计时
报告的是包装函数端到端墙钟平均时间，含分配及两次下发，不含JIT和输入传输，不伪称纯设备Kernel时间

## 固定提交参考依据

以下行号全部以`4cb292aec256a3bb3c0deea51c5cf55426b1f0e7`为准
为缩短长路径，仅在本引用表中定义前缀：`TH=csrc/ops/eagle_ffn/op_host/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/`，`TK=csrc/ops/eagle_ffn/op_kernel/3rd/mat_mul_v3/arch35/`，`CM=TK+cmct/`

[^H1]: `mindiesd/layers/eagle_ffn_linear.py:27-70`，`_chain_fallback/eagle_ffn_linear`，数学链、平台分支和接口
[^H2]: `mindiesd/layers/_custom_ops.py:701-748`，算子调用与fake；`csrc/plugin/register_ops.cpp:132,162`，schema及NPU注册
[^H3]: `csrc/plugin/eagle_ffn_linear.cpp:60-110`，`eagle_ffn_linear`，连续化、布局、输出和ACLNN调用
[^H4]: `csrc/ops/eagle_ffn/op_host/op_api/aclnn_eagle_ffn.cpp:1000-1067,1154-1208`，`GetFFNResultByL0Api/aclnnEagleFfnV2GetWorkspaceSize/aclnnEagleFfnV2`
[^H5]: `csrc/ops/eagle_ffn/op_host/op_api/ffn.cpp:67-92`，`l0op::EagleFfn`，输出分配与launcher；`csrc/ops/eagle_ffn/op_host/eagle_ffn_def.cpp:22-120`，注册dtype及平台
[^T0]: `csrc/ops/eagle_ffn/op_host/ffn_tiling.cpp:662-705`，`RunFusionKernelTiling`，A950路由及失败处理
[^T1]: `csrc/ops/eagle_ffn/op_host/ffn_arch35_tiling.cpp:40-125,160-410`，`FfnMatMulV3Tiling/FfnMatMulV3StreamKTiling/FFNArch35Tiling`，检查、两段策略、序列化、Key和workspace
[^T2]: `TH/matmul_v3_tiling_helper.cpp:45-79,125-149,191-270,366-529`，`CalL1TilingDefault/ResetBase/GetMaxBaseWithLimit/GetBalanceRateWithTail/GetBaseK/GetRebalanceBlock`；`TH/matmul_v3_common_advanced.h:22-57,170-190`，常量
[^T3]: `TH/matmul_v3_basic_aswt_tiling.cpp:45-103,138-280,318-347`，`CheckAL1FullLoad/CheckBL1FullLoad/DoAL1FullLoad/DoBL1FullLoad/DoOpTiling`
[^T4]: `TH/matmul_v3_asw_tiling.cpp:28-59`，`CalcTailBasicBlock/DoOpTiling`，尾轮拆分
[^T5]: `TH/matmul_v3_basic_streamk_tiling.cpp:31-81,139-211`，`IsCapable/DoOpTiling`及SK能力判断辅助函数
[^T6]: `TH/matmul_v3_base_tiling_advanced.h:99-105,426-474`，`AdjustOpTiling/GetTilingDataProcess`，基础块调整和字段转换
[^K1]: `csrc/ops/eagle_ffn/op_kernel/eagle_ffn_apt.cpp:31-82,86-230,235-376`，`MakeFfnMMTiling/RunFfn*/FfnArch35KernelImpl/eagle_ffn`
[^K2]: `CM/kernel/kernel_matmul_mix_without_que.h:94-139,165-206,235-256,284-354`，`Params/RunMmad/ProcessTiles/operator()`，UB直达、握手、半宽及实际核数
[^K3]: `TK/block_scheduler_aswt.h:33-165,197-242,300-407`，`BlockSchedulerAswtBuiltIn/GetBlockShape/GetBlockCoord/UpdateMNTileIdx`
[^K4]: `CM/block/block_mmad_builder.h:40-45,185-233`，`BlockMmadBuilder`；`CM/block/block_mmad_pingpong_without_que.h:163-235,243-425,587-613,703-829,1015-1105`，初始化、搬运、MMAD和全载
[^K5]: `TK/mat_mul_gelu_basic_cmct.h:13-18,54-135,138-272`，各mixed up的类型组合；`CM/epilogue/block_epilogue_elementwise.h:42-156`，`Init/DoFusionAndCast`
[^K6]: `CM/epilogue/fusion/fusion_regbase_act.h:35-85,179-258`，`GeluErfChunkB16/RegGeluErfB16/RegSwigluSingleB16`；`CM/epilogue/fusion/fusion_swiglu.h:35-110`，双上投影gate读取及激活
[^K7]: `TK/block_scheduler_streamk.h:25-216`，`BlockSchedulerStreamKBuiltIn`；`CM/kernel/kernel_matmul_streamk.h:310-410`，任务分配、scratch偏移和全局同步
[^K8]: `CM/block/block_mmad_streamk.h:384-466`，Bias加载和`NeedBias`；`CM/epilogue/block_epilogue_streamk.h:127-180`，`Run`，FP32归约和转换
[^K9]: `csrc/ops/eagle_ffn/op_kernel/ffn_arch35_tiling_key.h:22-64`，模板Key组合
[^K10]: `TK/mat_mul_pingpong_basic_cmct.h:25-70`，`MatMulActKernel`；`CM/kernel/kernel_matmul_without_que.h:201-302`，`operator()`，实际核数、全载预读和输出
