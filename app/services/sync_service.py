import os
import uuid
import logging
# ✅ Đổi: bỏ openai, dùng google-generativeai
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ✅ Đổi: cấu hình Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant_db:6333"))
COLLECTION_NAME = "ecommerce_products"

# Khởi tạo Text Splitter giống hệt luồng Batch
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

def sync_product_to_vector_db(product_id: int, name: str, description: str, category: str, price: float):
    """
    Hàm cập nhật hoặc thêm mới 1 sản phẩm vào Qdrant.
    Áp dụng nguyên tắc: Xóa sạch chunk cũ -> Tạo chunk mới -> Upsert.
    """
    doc_id = f"prod_{product_id}"
    logger.info(f"Đang đồng bộ sản phẩm {doc_id} lên Qdrant...")

    try:
        # 1. DỌN RÁC (DELETE OLD CHUNKS)
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="parent_doc_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                    ],
                )
            ),
        )
        logger.info(f"Đã dọn dẹp các chunk cũ của {doc_id}")

        safe_desc = description if description else "Không có mô tả chi tiết."
        content_text = f"Tên sản phẩm: {name}\nMô tả: {safe_desc}"

        chunks = text_splitter.split_text(content_text)
        points = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            enriched_text = f"[Thuộc sản phẩm ID: {doc_id}] {chunk_text}"

            # ✅ Đổi: gọi Gemini Embedding thay vì OpenAI
            embed_res = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=enriched_text,
                task_type="retrieval_document",
            )
            vector = embed_res["embedding"]

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "parent_doc_id": doc_id,
                        "content": enriched_text,
                        "category": category,
                        "price": float(price),
                        "type": "product_info"
                    }
                )
            )

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Đã Upsert thành công {len(points)} chunks mới cho sản phẩm {doc_id}.")

    except Exception as e:
        logger.error(f"Lỗi khi đồng bộ sản phẩm {doc_id}: {str(e)}")
        raise