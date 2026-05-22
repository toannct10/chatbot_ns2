import os
import uuid
import logging

from google import genai as genai_new
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.http import models

from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================
# LOGGER
# =========================================

logger = logging.getLogger(__name__)

# =========================================
# ENV
# =========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing")

if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL missing")

if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY missing")

# =========================================
# GEMINI
# =========================================

genai_client = genai_new.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# =========================================
# QDRANT CLOUD
# =========================================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "ecommerce_products"

# =========================================
# TEXT SPLITTER
# =========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

# =========================================
# SYNC SERVICE
# =========================================

def sync_service_to_vector_db(
    service_id: str,
    name: str,
    description: str,
    service_type: str,
    content_type: str,
):

    """
    Đồng bộ dữ liệu NextStep vào Qdrant Cloud
    """

    doc_id = f"service_{service_id}"

    logger.info(
        f"🔄 Đồng bộ {doc_id} lên Qdrant..."
    )

    try:

        # =====================================
        # DELETE OLD CHUNKS
        # =====================================

        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="parent_doc_id",
                            match=models.MatchValue(
                                value=doc_id
                            ),
                        ),
                    ],
                )
            ),
        )

        logger.info(
            f"🗑️ Đã xoá chunks cũ của {doc_id}"
        )

        # =====================================
        # PREPARE CONTENT
        # =====================================

        safe_desc = (
            description
            if description
            else "Không có mô tả."
        )

        content_text = (
            f"Tên: {name}\n"
            f"Nội dung: {safe_desc}"
        )

        chunks = text_splitter.split_text(
            content_text
        )

        logger.info(
            f"✂️ Created {len(chunks)} chunks"
        )

        points = []

        # =====================================
        # EMBED EACH CHUNK
        # =====================================

        for i, chunk_text in enumerate(chunks):

            chunk_id = f"{doc_id}_chunk_{i}"

            enriched_text = (
                f"[Thuộc dịch vụ: {name}] "
                f"{chunk_text}"
            )

            embed_res = (
                genai_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=enriched_text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
            )

            vector = embed_res.embeddings[0].values

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    chunk_id
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "parent_doc_id": doc_id,
                        "content": enriched_text,
                        "service_type": service_type,
                        "type": content_type,
                    }
                )
            )

        # =====================================
        # UPSERT
        # =====================================

        if points:

            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            logger.info(
                f"✅ Upsert thành công "
                f"{len(points)} chunks cho {doc_id}"
            )

    except Exception as e:

        logger.error(
            f"❌ Lỗi sync {doc_id}: {str(e)}"
        )

        raise

# =========================================
# KNOWLEDGE BASE
# =========================================

NEXTSTEP_KNOWLEDGE_BASE = [
    # giữ nguyên data của bạn
]

# =========================================
# SEED FUNCTION
# =========================================

def seed_knowledge_base():

    logger.info(
        "🚀 Bắt đầu seed dữ liệu NextStep..."
    )

    for item in NEXTSTEP_KNOWLEDGE_BASE:

        sync_service_to_vector_db(
            service_id=item["id"],
            name=item["name"],
            description=item["description"],
            service_type=item["service_type"],
            content_type=item["content_type"],
        )

    logger.info(
        "✅ Seed dữ liệu hoàn tất!"
    )