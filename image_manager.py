import os
import uuid
import torch
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
import hashlib

# ===================== 配置 =====================

IMAGE_DIR = "./images"
CHROMA_DIR = "./chroma_img_db"

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/data/linfengyun/models/sentence-transformers/clip-ViT-B-32"  # CLIP 图文模型

# ===================== 加载 CLIP =====================

model = SentenceTransformer(MODEL_PATH, device=device)

# ===================== ChromaDB =====================

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="image_search",
    metadata={"hnsw:space": "cosine"}
)

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ===================== 图像入库 =====================

def add_images(image_dir):
    for file in os.listdir(image_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(image_dir, file)
        image = Image.open(path).convert("RGB")

        # CLIP 图像嵌入
        img_emb = model.encode(image).tolist()
        img_id = file_hash(path)

        collection.add(
            ids=[img_id],
            embeddings=[img_emb],
            metadatas=[{
                "filename": file,
                "path": path
            }]
        )

        print(f"[入库] {file}")

# ===================== 文本搜图 =====================

def search_image(query, top_k=1):
    q_emb = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    hits = []
    for meta in results["metadatas"][0]:
        hits.append(meta["filename"])

    return hits

# ===================== 主入口 =====================

if __name__ == "__main__":

    # 一次性入库（首次运行）
    add_images(IMAGE_DIR)

    # 文本查询
    query = "sunset at the beach"
    results = search_image(query)

    print("\n最匹配的图片：")
    for r in results:
        print("-", r)
