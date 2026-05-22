import logging
from fastapi import APIRouter, HTTPException
from app.services.sync_service import sync_service_to_vector_db, seed_knowledge_base
 
logger = logging.getLogger(__name__)
router = APIRouter()
 
 
@router.post("/service/sync", summary="Thêm/Cập nhật nội dung dịch vụ vào Vector DB")
async def sync_service(
    service_id: str,
    name: str,
    description: str,
    service_type: str,
    content_type: str,
):
    """
    Đồng bộ 1 nội dung dịch vụ NextStep lên Qdrant.
    - service_type: general | deepscan_cv | mooc_interview | pricing | payment | support
    - content_type: service_info | pricing | faq
    """
    try:
        sync_service_to_vector_db(
            service_id=service_id,
            name=name,
            description=description,
            service_type=service_type,
            content_type=content_type,
        )
        return {
            "status": "success",
            "message": f"Nội dung '{name}' đã được đồng bộ lên Qdrant!"
        }
    except Exception as e:
        logger.error(f"Lỗi API sync service: {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý dữ liệu. Vui lòng kiểm tra log.")
 
 
@router.post("/service/seed", summary="Seed toàn bộ dữ liệu NextStep vào Vector DB")
async def seed_data():
    """
    Chạy 1 lần để đẩy toàn bộ knowledge base NextStep vào Qdrant.
    """
    try:
        seed_knowledge_base()
        return {
            "status": "success",
            "message": "Đã seed toàn bộ dữ liệu NextStep vào Qdrant!"
        }
    except Exception as e:
        logger.error(f"Lỗi seed data: {e}")
        raise HTTPException(status_code=500, detail="Lỗi seed dữ liệu. Vui lòng kiểm tra log.")