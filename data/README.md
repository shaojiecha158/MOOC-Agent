# 数据说明 (Data Description)

由于 MOOCCube 原始数据集体积较大，本项目仓库 **不包含** 完整的原始数据文件。

此处仅提供：
1. **数据处理脚本** (`process_relations.py`): 用于处理原始数据、构建图谱路径并生成指令微调数据。
2. **微调数据样例** (`sample_data.json`): 展示处理后的指令微调数据格式（Instruction/Input/Output）。

如果您需要复现完整的数据生成过程，请访问 [MOOCCube 官网](http://mooccube.com/) 下载原始数据集，并将其解压至本目录。