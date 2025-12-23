# MOOC-Agent: Knowledge-Graph-Augmented LLM for Course Recommendation
# 基于知识图谱的垂类教育大模型 Agent

<div align="center">
  <img src="assets/demo_screenshot.png" width="80%" alt="Demo Screenshot">
</div>

## 📖 项目简介 (Introduction)
本项目探索了如何将 **MOOCCube 知识图谱** 注入到通用大模型（Qwen2.5-7B）中，构建一个具备逻辑推理能力的智能教育规划 Agent。
针对通用模型在垂直领域推荐中存在的“幻觉”和“缺乏逻辑链条”问题，本项目通过 NetworkX 构建课程先修路径，利用 **LoRA** 技术进行指令微调（SFT），实现了基于先修关系的课程路径规划。

## ✨ 核心特性 (Key Features)
- **实体落地 (Entity Grounding)**: 能够输出带有 `(自主模式)` 后缀的真实 MOOC 课程，消除了基座模型编造课程名的幻觉。
- **双重推理模式**: 
  - **逻辑连贯**: 针对有先修关系的课程，解释知识点依赖（如 C语言 -> 数据结构）。
  - **内容推荐**: 针对跨域/无连边课程，基于内容语义进行推荐。
- **格式化输出**: 严格遵循结构化指令，便于下游系统集成。

## 🏗️ 技术架构 (Methodology)
1. **Graph Construction**: 基于 MOOCCube 数据集，利用 `NetworkX` 构建包含 700+ 概念与课程的依赖图谱。
2. **Data Generation**: 设计基于图谱游走（Graph Walk）的算法，构建了 190k 条指令微调数据 (`train_data.json`)。
3. **Efficient Fine-tuning**: 使用 `LLaMA-Factory` 框架，采用 LoRA (Rank=32, Alpha=64) 对 `Qwen2.5-7B-Instruct` 进行微调。
4. **Inference Optimization**: 引入 `Repetition Penalty` 和 `Temperature Scaling` 抑制模型重复生成，提升语言多样性。

## 📊 效果对比 (Case Study)

| 用户输入 | 基座模型 (Qwen2.5-Base) | MOOC-Agent (Ours) |
| :--- | :--- | :--- |
| **已掌握[C语言]，目标[数据结构]** | 推荐《Python爬虫》(幻觉)，理由泛泛而谈。 | **推荐《操作系统(自主模式)》**。理由：基于先修关系，解释了从编程到系统的逻辑。 |
| **已掌握[英语]，目标[商务翻译]** | 罗列了一堆通用建议。 | 识别为无先修关系，自动切换至**【内容推荐】**模式，推荐相关阅读课程。 |

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖

pip install -r requirements.txt

### 2. 下载模型权重
本模型的 LoRA 权重已发布至 ModelScope 社区，可直接下载：

- **ModelScope 模型主页**: [点击跳转](https://www.modelscope.cn/models/shaojiecha/MOOC-Agent-LoRA)

或者使用 Python 代码下载：
```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('shaojiecha/MOOC-Agent-LoRA')