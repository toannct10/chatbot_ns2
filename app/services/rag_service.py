import os
import json
import logging
from google import genai as genai_new
import google.generativeai as genai  # giữ lại cho GenerativeModel
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
 
logger = logging.getLogger(__name__)
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant_db:6333")
 
genai.configure(api_key=GEMINI_API_KEY)
genai_client = genai_new.Client(api_key=GEMINI_API_KEY)  # thêm dòng này
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
COLLECTION_NAME = "ecommerce_products"
 
def retrieve_context(query: str, service_type: str = None, top_k: int = 4) -> str:
    try:
        embed_res = genai_client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = embed_res.embeddings[0].values
 
        must_conditions = []
 
        if service_type:
            must_conditions.append(
                models.FieldCondition(
                    key="service_type",
                    match=models.MatchValue(value=service_type)
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
                "service_type": point.payload.get("service_type", "general"),
                "type": point.payload.get("type", "service_info"),
                "score": point.score,
                "metadata": point.payload
            }
            for point in search_results.points
        ]
 
        logger.info("Context chunks: %s", context_chunks)
 
        formatted_contexts = []
        for chunk in context_chunks:
            if chunk["type"] == "faq":
                combined_text = f"[CÂU HỎI THƯỜNG GẶP]\n{chunk['content']}"
            elif chunk["type"] == "pricing":
                combined_text = f"[THÔNG TIN GÓI DỊCH VỤ]\n{chunk['content']}"
            else:
                combined_text = f"[THÔNG TIN DỊCH VỤ]\n{chunk['content']}"
            formatted_contexts.append(combined_text)
 
        context_text = "\n\n" + "="*30 + "\n\n".join(formatted_contexts) + "\n\n" + "="*30
        logger.info(f"Context Text gửi cho LLM:\n{context_text}")
        return context_text
 
    except Exception as e:
        logger.error(f"❌ Lỗi khi truy xuất Qdrant: {e}")
        return ""
 
 
def analyze_user_query(query: str) -> dict:
    analyzer_prompt = """Bạn là chuyên gia phân tích câu hỏi về dịch vụ NextStep - nền tảng hỗ trợ người tìm việc.
        Hãy đọc câu hỏi và trả về ĐÚNG 1 ĐỊNH DẠNG JSON.
        1. "service_type": loại dịch vụ người dùng hỏi đến:
           - "deepscan_cv": hỏi về tính năng chấm điểm, đánh giá CV
           - "mooc_interview": hỏi về tính năng phỏng vấn mô phỏng với AI
           - "pricing": hỏi về giá cả, gói dịch vụ, chi phí
           - "payment": hỏi về thanh toán, VNPAY, MoMo
           - "support": hỏi về hỗ trợ, liên hệ, lỗi hệ thống
           - null: nếu câu hỏi chung hoặc không rõ thuộc loại nào
        2. "intent": mục đích câu hỏi:
           - "feature_info": muốn biết tính năng hoạt động như thế nào
           - "pricing_info": muốn biết giá/gói dịch vụ
           - "how_to": muốn biết cách sử dụng
           - "troubleshoot": gặp lỗi hoặc vấn đề
           - "general": câu hỏi chung về NextStep
 
        Trả về duy nhất JSON, không thêm bất kỳ text nào khác.
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
        return extracted_data
 
    except Exception as e:
        logger.error(f"Lỗi phân tích query: {e}")
        return {"service_type": None, "intent": "general"}
 
 
def generate_answer_stream(query: str, service_type: str = None, **kwargs):
    context = retrieve_context(query, service_type)
 
    if not context:
        yield "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp với câu hỏi của bạn. Bạn có thể liên hệ hỗ trợ qua email: ngcongtoan10@gmail.com"
        return
 
    system_prompt = """Bạn là trợ lý AI chuyên nghiệp của NextStep - nền tảng hỗ trợ người tìm việc cải thiện CV và kỹ năng phỏng vấn thông qua trí tuệ nhân tạo.
 
        THÔNG TIN VỀ NEXTSTEP:
        - DeepScan CV: Chấm điểm và đánh giá CV dựa trên Job Description
        - Mooc Interview AI: Mô phỏng phỏng vấn 1-1 với chuyên gia HR AI
        - Hỗ trợ thanh toán: VNPAY và MoMo
        - Email hỗ trợ: ngcongtoan10@gmail.com
 
        QUY TẮC BẮT BUỘC:
        1. Chỉ tư vấn về dịch vụ của NextStep (CV, phỏng vấn, tài khoản, thanh toán). Không trả lời các chủ đề không liên quan.
        2. Thông tin trong [NGỮ CẢNH DỊCH VỤ] là dữ liệu chính xác — hãy dựa vào đó để trả lời.
        3. Trả lời thân thiện, ngắn gọn, dễ hiểu. Khuyến khích người dùng trải nghiệm dịch vụ.
        4. Nếu câu hỏi nằm ngoài phạm vi dịch vụ NextStep, lịch sự từ chối và hướng dẫn liên hệ hỗ trợ.
        5. Nếu [NGỮ CẢNH DỊCH VỤ] không đủ thông tin, hướng người dùng liên hệ email hỗ trợ.
 
        [NGỮ CẢNH DỊCH VỤ]:
        {context_data}
        """
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_CHAT_MODEL,
            system_instruction=system_prompt.format(context_data=context),
            generation_config=genai.GenerationConfig(
                temperature=0.1,
            ),
        )
        response = model.generate_content(query, stream=True)
 
        for chunk in response:
            if chunk.text:
                yield chunk.text
 
    except Exception as e:
        logger.error(f"❌ Lỗi khi gọi Gemini API: {e}")
        yield "Hệ thống AI đang quá tải, vui lòng thử lại sau giây lát."