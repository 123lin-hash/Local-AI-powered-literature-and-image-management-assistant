from sentence_transformers import SentenceTransformer, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import fitz
import os
import shutil
import uuid
import hashlib
import torch

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_PATH = "/data/linfengyun/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(EMBED_MODEL_PATH, device=device)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)

CHROMA_DIR = "./chroma_db"
PAPER_ROOT = "./papers"
SORTED_ROOT = "./sorted_papers"

# 主题定义
TOPICS = {
    "CV": "computer vision, image understanding, object detection,image segmentation, visual representation, vision transformer, image classification, visual foundation model",
    "NLP": "natural language processing, large language model, language model, token, text generation, pretraining, instruction tuning",
    "RL": "reinforcement learning, policy optimization, reward model, preference learning, agent learning"
}

# ChromaDB

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"}
)


semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

#PDF 处理

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []

    for page in doc:
        texts.append(page.get_text())

    return "\n".join(texts)

def semantic_chunk_pdf(pdf_path):
    raw_text = extract_pdf_text(pdf_path)
    docs = [Document(page_content=raw_text)]
    chunks = semantic_splitter.split_documents(docs)
    return chunks  # List[Document]

#自动分类

def classify_text(text):
    paper_emb = embedder.encode(text, convert_to_tensor=True)

    topic_names = list(TOPICS.keys())
    topic_descs = list(TOPICS.values())
    topic_embs = embedder.encode(topic_descs, convert_to_tensor=True)

    sims = util.cos_sim(paper_emb, topic_embs)[0]
    idx = sims.argmax().item()

    return topic_names[idx], sims[idx].item()

#入库 & 自动整理

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def add_paper(pdf_path, move_file=True):
    pdf_id = file_hash(pdf_path)
    chunks = semantic_chunk_pdf(pdf_path)

    if not chunks:
        print(f"[跳过] 无法解析: {pdf_path}")
        return

    full_text = " ".join([c.page_content for c in chunks])
    topic, score = classify_text(full_text)

    # 文件归档
    if move_file:
        target_dir = os.path.join(SORTED_ROOT, topic)
        os.makedirs(target_dir, exist_ok=True)
        new_path = os.path.join(target_dir, os.path.basename(pdf_path))
        shutil.move(pdf_path, new_path)
    else:
        new_path = pdf_path

    # 语义块入库
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{pdf_id}_{idx}"   
        embedding = embedder.encode(chunk.page_content).tolist()

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.page_content],
            metadatas=[{
                "pdf_id": pdf_id,
                "filename": os.path.basename(new_path),
                "path": new_path,
                "topic": topic,
                "chunk_id": idx
            }]
        )

    print(f"[入库] {os.path.basename(pdf_path)} → {topic}")

def batch_sort(pdf_dir): 
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            try:
                add_paper(os.path.join(pdf_dir, file))
            except Exception as e:
                print(f"[失败] {file}: {e}")

# 语义搜索

def search_paper(query, top_k=1):
    q_emb = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    files = []
    for meta in results["metadatas"][0]:
        files.append(meta["filename"])

    return list(dict.fromkeys(files))

