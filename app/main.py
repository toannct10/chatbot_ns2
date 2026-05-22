from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat_routers import router as chat_router
from app.api.admin_routers import router as product_router
import logging

# Cấu hình logging chuẩn cho toàn bộ app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="E-commerce RAG API",
    description="Hệ thống trợ lý ảo thông minh cho E-commerce sử dụng Qdrant và OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["Chatbot"])
app.include_router(product_router, prefix="/api", tags=["Product"])

# Health Check
@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "message": "Chào mừng đến với E-commerce RAG API! Truy cập /docs để xem tài liệu API."
    }

logger.info("🚀 Server FastAPI đã khởi động thành công!")