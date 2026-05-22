from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging

# Import hàm sinh câu trả lời từ tầng Service mà ta vừa viết
from app.services.rag_service import generate_answer_stream, analyze_user_query
from app.schemas.chat_schema import ChatRequest

logger = logging.getLogger(__name__)

# init router
router = APIRouter()

@router.post("/chat", summary="Chat với Trợ lý ảo E-commerce")
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint nhận câu hỏi và trả về câu trả lời dạng Stream (SSE).
    Hỗ trợ lọc trước theo danh mục và giá tiền để tăng độ chính xác.
    """
    try:
        filters = analyze_user_query(request.query)
        answer_generator = generate_answer_stream(
            query=request.query,
            category=filters.get("category"),
            max_price=filters.get("max_price")
        )
        return StreamingResponse(
            answer_generator, 
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"❌ Lỗi tại endpoint /chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống nội bộ. Vui lòng thử lại sau.")