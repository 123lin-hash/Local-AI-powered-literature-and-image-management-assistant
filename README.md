# 本地 AI 智能文献与图像管理助手
## 1.项目简介
本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现文献管理和图像管理。
## 2.核心功能列表
### 2.1 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统需基于语义理解返回最相关的论文文件，进阶要求可返回具体的论文片段或页码。
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
    *   **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。
*   **文件索引**: 支持仅返回相关文件列表，方便快速定位所需文献。

### 2.2 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。
## 3.环境配置
- 操作系统：Ubuntu 20.04
- Python 版本：Python 3.10
- 虚拟环境：conda
- GPU：RTX 3090,24G
- CUDA：12.2
## 4.依赖安装说明
- torch
- torchvision
- torchaudio
- sentence-transformers
- chromadb
- langchain
- langchain-experimental
- langchain-core
- langchain-huggingface
- pymupdf
- pillow
- modelscope
## 5.详细的使用说明
### 5.1 项目结构
```text
.
├── main.py                # 统一命令行入口
├── paper_manager.py       # 论文管理与搜索模块
├── image_manager.py       # 图片入库与以文搜图模块
├── papers/                # 原始论文 PDF
├── sorted_papers/         # 自动分类后的论文目录
├── images/                # 图片目录
├── chroma_db/             # 论文向量数据库（自动生成）
└── chroma_img_db/         # 图片向量数据库（自动生成）
```
### 5.2 命令行示例
#### 添加单个论文并分类
```bash
python main.py add_paper ./papers/Attention-is-all-you-need.pdf
```
#### 批量添加并分类整个目录下的论文
```bash
python main.py add_paper ./papers
```
#### 返回最相关论文
```bash
python main.py search_paper "什么是位置编码"
```
#### 搜索最匹配的图片
```bash
python main.py search_image "海边的日落"
```
## 6.技术选型说明
*   **文本嵌入**: `paraphrase-multilingual-MiniLM-L12-v2` —— 多语言支持（覆盖超过50种语言，如英语、中文、阿拉伯语和保加利亚语）、高效计算资源利用（基于MiniLM架构，参数量较小，适合资源受限环境）以及强大的泛化能力（在未见过的数据上表现良好）。‌
*   **图像嵌入**: `ViT-B-32` —— OpenAI 开源的经典图文匹配模型。
*   **向量数据库**: `ChromaDB` —— 无需服务器，开箱即用的嵌入式数据库。