import os
import json
import logging

from google import genai as genai_new
import google.generativeai as genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.http import models

# =========================================
# LOGGER
# =========================================

logger = logging.getLogger(__name__)

# =========================================
# ENV VARIABLES
# =========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# =========================================
# VALIDATE ENV
# =========================================

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing")

if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL is missing")

if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY is missing")

# =========================================
# GEMINI CONFIG
# =========================================

genai.configure(api_key=GEMINI_API_KEY)

genai_client = genai_new.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-3.1-flash"

# =========================================
# QDRANT CLOUD
# =========================================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "ecommerce_products"

# =========================================
# RETRIEVE CONTEXT
# =========================================

def retrieve_context(
    query: str,
    service_type: str = None,
    top_k: int = 4
) -> str:

    try:

        logger.info(f"🔍 Query: {query}")

        # =====================================
        # EMBEDDING QUERY
        # =====================================

        embed_res = genai_client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        query_vector = embed_res.embeddings[0].values

        logger.info("✅ Query embedding created")

        # =====================================
        # FILTER
        # =====================================

        must_conditions = []

        if service_type:
            must_conditions.append(
                models.FieldCondition(
                    key="service_type",
                    match=models.MatchValue(value=service_type)
                )
            )

        search_filter = (
            models.Filter(must=must_conditions)
            if must_conditions
            else None
        )

        # =====================================
        # SEARCH QDRANT
        # =====================================

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k
        )

        logger.info(f"✅ Retrieved {len(search_results.points)} chunks")

        # DEBUG RESULTS
        logger.info(f"DEBUG RESULTS: {search_results.points}")

        if not search_results.points:
            logger.warning("⚠️ No matching context found")
            return ""

        # =====================================
        # FORMAT CHUNKS
        # =====================================

        context_chunks = [
            {
                "content": point.payload.get("content", ""),
                "service_type": point.payload.get(
                    "service_type",
                    "general"
                ),
                "type": point.payload.get(
                    "type",
                    "service_info"
                ),
                "score": point.score,
                "metadata": point.payload
            }
            for point in search_results.points
        ]

        logger.info("Context chunks: %s", context_chunks)

        formatted_contexts = []

        for chunk in context_chunks:

            if chunk["type"] == "faq":
                combined_text = (
                    f"[CÂU HỎI THƯỜNG GẶP]\n"
                    f"{chunk['content']}"
                )

            elif chunk["type"] == "pricing":
                combined_text = (
                    f"[THÔNG TIN GÓI DỊCH VỤ]\n"
                    f"{chunk['content']}"
                )

            else:
                combined_text = (
                    f"[THÔNG TIN DỊCH VỤ]\n"
                    f"{chunk['content']}"
                )

            formatted_contexts.append(combined_text)

        context_text = (
            "\n\n"
            + "=" * 30
            + "\n\n".join(formatted_contexts)
            + "\n\n"
            + "=" * 30
        )

        logger.info(
            f"📄 Context gửi cho LLM:\n{context_text}"
        )

        return context_text

    except Exception as e:

        logger.error(
            f"❌ Lỗi khi truy xuất Qdrant: {e}"
        )

        return ""

# =========================================
# ANALYZE USER QUERY
# =========================================

def analyze_user_query(query: str) -> dict:

    analyzer_prompt = """
Bạn là chuyên gia phân tích câu hỏi về dịch vụ NextStep.

Trả về JSON duy nhất.

{
  "service_type": "...",
  "intent": "..."
}
"""

    try:

        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=analyzer_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        response = model.generate_content(query)

        extracted_data = json.loads(response.text)

        logger.info(
            f"✅ Query analysis: {extracted_data}"
        )

        return extracted_data

    except Exception as e:

        logger.error(
            f"❌ Lỗi phân tích query: {e}"
        )

        return {
            "service_type": None,
            "intent": "general"
        }

# =========================================
# GENERATE ANSWER
# =========================================

def generate_answer_stream(
    query: str,
    service_type: str = None,
    **kwargs
):

    context = retrieve_context(
        query,
        service_type
    )

    if not context:

        yield (
            "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp "
            "về dịch vụ NextStep. "
            "Vui lòng liên hệ hỗ trợ qua email: "
            "ngcongtoan10@gmail.com"
        )

        return

    system_prompt = """
Bạn là trợ lý AI chuyên nghiệp của NextStep.

Chỉ trả lời dựa trên context được cung cấp.

Nếu thiếu thông tin:
- lịch sự xin lỗi
- hướng dẫn liên hệ hỗ trợ

[NGỮ CẢNH DỊCH VỤ]:
{context_data}
"""

    try:

        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=system_prompt.format(
                context_data=context
            ),
            generation_config=genai.GenerationConfig(
                temperature=0.1,
            ),
        )

        response = model.generate_content(
            query,
            stream=True
        )

        for chunk in response:

            if chunk.text:
                yield chunk.text

    except Exception as e:

        logger.error(
            f"❌ Lỗi khi gọi Gemini API: {e}"
        )

        yield (
            "Hệ thống AI đang quá tải, "
            "vui lòng thử lại sau."
        )