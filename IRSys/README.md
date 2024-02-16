# Summary
主要整理业界与学术界的information retrieval方法，聚焦match/pre-rank。
| period of surveyed | number of papers| companies |  Match-related paper num | pre-rank related paper num|
| :---    | :---:  | :--- | :--- | :--- |
| 2021-2023| 92 | Alibaba, ByteDance, Facebook, Google, Baidu, Weixin, Pinterest, Shoppe, Amazon, etc.| 79 | 12 |

**注**：pre-rank里面包含了一些多目标的论文，即可用于精排也可用于粗排。

会议分布
|venue name| number of papers |
| :---    | :---:  |
| KDD| 19 |
| RecSys|4|
| CIKM| 21|
| SIGIR| 12|
| VLDB|3|
| WSDM|4|
| WWW|14|
| NIPS| 3|
| ICLR|1|
| ICML| 1|
| SIGKDD| 1|
| AAAI| 2 |
| DLP-KDD|1|
|IJCAI| 1|
| DSAFAA|1|
| arXiv|3|

公司分布
|venue name| number of papers |
| :---    | :---:  |
| Alibaba| 30 |
| ByteDance|2|
| Facebook(Meta)| 4|
| Google| 5|
| Baidu|3|
| Tencent(Weixin)|4|
| Pinterest|6|
| Shoppe|2 |
| Twitter| 2|
| Amazon|4|
| SalesForce|1|
| Huawei|4|
|microsoft| 4|
|JD| 3|
|Meituan|2|
|Adobe|1|
|Kuaishou| 1 |
|NetEease|1|
|academia|12|

**注**:本次survey重点考察近三年的技术发展，但也会根据需要callback这些技术的前身
# Table of Contents
1. [Rules](#rules)
2. [Alibaba Summary](#alibaba)
    1. [Match Paper Snapshots](#alimatch)
    2. [Pre-rank Paper Snapshots](#aliprerank)
3. [Others Summary](#others)
    1. [Match Paper Snapshots](#othermatch)

<a id="rules"></a>
## Rules
将文章分为三类：
- 架构（涉及ANN，整个训练及serving框架的改动）
- 表征（网络结构)
- 特征（特征交叉，图数据应用）

<a id="alibaba"></a>
## Alibaba Summary
| Company | Stage | period of surveyed | number of papers |
| :---    | :---  | :---               | :---             |
| alibaba | match | 2021-2023          | 22   |
| alibaba | pre-rank | 2021-2023          |      8       |

<a id="alimatch"></a>
#### Match Paper Snapshots
| title | 分类 | 解决问题 | 备注 |
| :---    | :---:  | :--- | :--- |
|[2018]TDM [2019]JTM [2020]OTM [2021]POEEM [2022]二向箔| 架构 | 召回常用embedding-based的方法，但是常用的召回方式限制了模型的表达能力 | [2018]TDM/[2019]JTM/[2020]OTM使用EM的方式联合训练模型和Index，[2021]POEEM提出感知训练的PQ index，[2022]二向箔在模型训练中加入auxiliary loss使得训练出来的embedding仍可以使用两段式的方法构建Index |
|[2019]MIND [2019]SDM [2020]ComiRec [2021]SINE [2021]MGDSPR [2021]PDN [2022]ADI [2022]XDM | 表征 | 召回模型在捕捉用户兴趣多样性上表现欠佳 | 将推荐建模为预测用户的下一个行为。[2019]SDM探索长短期用户行为表征，[2021]MGDSPR为搜索提供了embedding-based召回解决方案，[2021]PDN使用多层网络发掘用户和可能相关的item之间的相关度，[2019]MIND使用胶囊学习用户多兴趣embedding，[2019]SINE探索不基于胶囊的多兴趣embedding，[2020]ComiRec和[2022]ADI关心cross-domain Rec问题，[2022]XDM认为曝光未点击的item的相关度介于点击与完全随机的item之间，使用metric learning将未点击item的embedding映射到合理的向量空间取得优于SDM的召回效果 |
|[2019]IntentGC [2020]M2GRL [2020]Swing&Surprise [2021]HetMatch [2022]DC-GNN [2023]CC-GCN| 特征 | 召回环节数据稀疏 |[2019]IntentGC不仅使用点击，而且使用auxiliary关系挖掘更丰富的信息，用GCN学习heterogeneous relations，[2020]Swing&Surprise提出一个进使用用户行为和类目信息来进行相死品和搭配品推荐的I2I推荐方法，[2021]HetMatch用于为B端广告主推送keyword，把keyword，ad，item当作node以点击关系建立异构图，[2022]DC-GNN以HetMatch类似的方式建立异构图，在推理网络结构轻量化和预训练上做了创新，[2023]CC-GCN使用全新的方式构建异构图，更加能抓住语意信息，并且使用虚构数据提高模型对长尾query以及长尾item的召回效果|

<a id="aliprerank"></a>
#### Pre-rank Paper Snapshots
| title | 分类 | 解决问题 | 备注 |
| :---    | :---:  | :--- | :--- |
|[2021]CAN [2021]PCF-GNN| 特征 | 显式交叉特征实验效果优于隐式交叉，但是显式交叉的特征空间是特征数量的笛卡尔积，实际落地有困难| [2021]CAN提出feed+induction的模式来表达显式交叉特征A+B:A的embedding通过B特有的网络。在电商实际落地中，A即为用户的行为序列，B为target id以及user的侧信息，在标准数据集上CAN的效果优于显式交叉特征，[2021]PCF-GNN利用graph-agg来构建交叉特征|
| [2022]MIM-DRCFR [2023]DTRN| 表征 | 粗排需要精确评估广告的效果，以及常常需要准确评估多个目标。预训练也常被使用 | [2022]MIM-DRCFR在causal inference的框架推理广告在个体粒度的表现ITE(individual treatment effect)，，这种方式可以相对准确地评估intervention的uplift，在message push的online A/B test中，daily login users after push有显著提升，[2023]DTRN使用hypernets和contional transformer产出task-specific and interest-specific的embedding，不同于之前其他公用backbone的MTL，DTRN的方式相当于对公用的backbone进行了task-specific的修饰，在多个MTL数据上超越DIEN，DIN等方法|
|[2020]COLD [2022]KEEP [2023]COPR [2023]ASMOL| 架构 | pre-rank需要关心与上游match、下游rank的联动以及其功能定位 | [2020]COLD提出一系列工程优化提升粗排的对算力的使用能力，模糊化粗排和精排模型结构的界限，[2022]KEEP提出一个可以插拔的外接知识图谱缓解训练时的数据稀疏，[2023COPR]重构了pre-rank的模型训练方式：将rank list按照ecpm降序排列并分成n段，模型目标拆分成两个，一个是评估段之间广告的排序，一个是精确评估广告的pctr|

<a id="others"></a>
## Others Summary
**注** 本次survey重点考察近三年的技术发展，但也会根据需要callback这些技术的前身
| Stage | period of surveyed | number of papers |
| :---  | :---               | :---             |
| match | 2021-2023          | 57   |
| pre-rank | 2021-2023          | 4   |

<a id="othermatch"></a>
#### Match Paper Snapshots
| title | 分类 | 解决问题 | 备注 |
| :---    | :---:  | :--- | :--- |
|[2019]MOBIUS [2020]EBR [2021]DR [2021]SSL [2023]MPKG [2023]LightSAGE [2023]CIGF| 架构|||
|[2020]SimClusters [2020]NIA-GCN [2021]SGL [2022]NAVIP [2022]SpectralGraphEmb [2022]MultiBiStage [2022]TwHIN [2022]SVD-GCN [2023]SimEmb| 表征 |||
|[2019]Correct-sfx [2020]SampleOptimization [2020]UTPM [2023]IDW| 特征 |||
