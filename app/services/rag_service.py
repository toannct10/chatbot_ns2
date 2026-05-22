import os
import json
import logging
# ✅ Đổi: bỏ openai, dùng google-generativeai
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

# ✅ Đổi: cấu hình Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant_db:6333")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_CHAT_MODEL = "gemini-1.5-flash"

qdrant = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "ecommerce_products"


def retrieve_context(query: str, category: str = None, max_price: float = None, top_k: int = 4) -> str:
    try:
        # ✅ Đổi: Gemini embedding, task_type="retrieval_query" cho phía search
        embed_res = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        query_vector = embed_res["embedding"]

        must_conditions = []

        if category:
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            )

        if max_price:
            must_conditions.append(
                models.FieldCondition(
                    key="price",
                    range=models.Range(lte=max_price)
                )
            )

        search_filter = models.Filter(must=must_conditions) if must_conditions else None

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k
        )

        if not search_results.points:
            return ""

        context_chunks = [
            {
                "content": point.payload.get("content", ""),
                "price": point.payload.get("price", 0),
                "type": point.payload.get("type", "product_info"),
                "score": point.score,
                "metadata": point.payload
            }
            for point in search_results.points
        ]

        logger.info("Context chunks: %s", context_chunks)

        formatted_contexts = []
        for chunk in context_chunks:
            if chunk['type'] == 'policy':
                combined_text = f"[CHÍNH SÁCH CỬA HÀNG]\n{chunk['content']}"
            else:
                formatted_price = f"{chunk['price']:,.0f}".replace(",", ".")
                combined_text = f"[SẢN PHẨM]\n{chunk['content']}\nGiá bán: {formatted_price} VNĐ"
            formatted_contexts.append(combined_text)

        context_text = "\n\n" + "="*30 + "\n\n".join(formatted_contexts) + "\n\n" + "="*30
        logger.info(f"Context Text gửi cho LLM:\n{context_text}")
        return context_text

    except Exception as e:
        logger.error(f"❌ Lỗi khi truy xuất Qdrant: {e}")
        return ""


def analyze_user_query(query: str) -> dict:
    analyzer_prompt = """Bạn là chuyên gia trích xuất dữ liệu. Hãy đọc câu hỏi và trả về ĐÚNG 1 ĐỊNH DẠNG JSON.
        1. "category": "dien_tu" (điện thoại, tai nghe...), "thoi_trang" (quần áo, balo...), hoặc null nếu không rõ.
        2. "max_price": CHÚ Ý - Phải dịch các từ chỉ tiền tệ sang số nguyên VNĐ.
        - Ví dụ: "10 triệu", "10 củ" -> 10000000
        - Ví dụ: "500k", "500 cành" -> 500000
        - Ví dụ: "dưới 2 triệu" -> 2000000
        - Nếu câu hỏi KHÔNG nhắc đến giới hạn giá tối đa -> null

        Trả về duy nhất JSON, không thêm bất kỳ text nào khác.
        """
    try:
        # ✅ Đổi: Gemini với response_mime_type="application/json" thay vì gpt-4o-mini
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
        return extracted_data

    except Exception as e:
        logger.error(f"Lỗi phân tích query: {e}")
        return {"category": None, "max_price": None}


def generate_answer_stream(query: str, category: str = None, max_price: float = None):
    context = retrieve_context(query, category, max_price)

    if not context:
        yield "Xin lỗi, hiện tại cửa hàng không tìm thấy sản phẩm hoặc thông tin nào phù hợp với yêu cầu của bạn."
        return

    system_prompt = """Bạn là trợ lý ảo AI xuất sắc của hệ thống E-commerce.
        QUY TẮC BẮT BUỘC:
        1. Thông tin trong [NGỮ CẢNH SẢN PHẨM] là các sản phẩm ĐÃ ĐƯỢC HỆ THỐNG LỌC CHUẨN XÁC theo mức giá và danh mục khách yêu cầu. 
        2. Hãy TỰ TIN giới thiệu các sản phẩm này. TUYỆT ĐỐI KHÔNG được nói là "không có sản phẩm nào phù hợp" nếu trong ngữ cảnh có chứa sản phẩm.
        3. KHÔNG tự ý so sánh toán học (lớn hơn, nhỏ hơn). Chỉ trình bày lại tên, mô tả và giá tiền của sản phẩm trong ngữ cảnh một cách thân thiện, hấp dẫn để chốt sale.
        4. Nếu [NGỮ CẢNH SẢN PHẨM] hoàn toàn trống, lúc đó mới lịch sự xin lỗi khách hàng.

        [NGỮ CẢNH SẢN PHẨM]:
        {context_data}
        """
    try:
        # ✅ Đổi: Gemini stream thay vì OpenAI stream
        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=system_prompt.format(context_data=context),
            generation_config=genai.GenerationConfig(
                temperature=0.1,
            ),
        )
        response = model.generate_content(query, stream=True)

        # ✅ Đổi: chunk.text thay vì chunk.choices[0].delta.content
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        logger.error(f"❌ Lỗi khi gọi Gemini API: {e}")
        yield "Hệ thống AI đang quá tải, vui lòng thử lại sau giây lát."