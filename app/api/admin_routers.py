import os
import pymysql
pymysql.install_as_MySQLdb()
from fastapi import APIRouter, HTTPException
import logging
from sqlalchemy import create_engine, text
from app.services.sync_service import sync_product_to_vector_db
from app.schemas.product_schema import ProductUpdatePayload

logger = logging.getLogger(__name__)
router = APIRouter()

DB_URL = os.getenv("DATABASE_URL")

# ✅ SSL riêng qua connect_args, không để trong URL
engine = create_engine(
    DB_URL,
    connect_args={
        "ssl": {"ssl_mode": "REQUIRED"}
    }
)
def upsert_product_to_mysql(payload: ProductUpdatePayload):
    """
    Lưu hoặc cập nhật sản phẩm vào MySQL (Source of Truth)
    """
    upsert_query = text("""
        INSERT INTO products (product_id, name, description, category, price, status)
        VALUES (:product_id, :name, :description, :category, :price, 'active')
        ON DUPLICATE KEY UPDATE 
            name = VALUES(name),
            description = VALUES(description),
            category = VALUES(category),
            price = VALUES(price),
            updated_at = CURRENT_TIMESTAMP;
    """)
    
    with engine.begin() as conn: # Dùng begin() để tự động commit transaction
        conn.execute(upsert_query, {
            "product_id": payload.product_id,
            "name": payload.name,
            "description": payload.description,
            "category": payload.category,
            "price": payload.price
        })
        logger.info(f"Đã lưu thành công sản phẩm {payload.product_id} vào MySQL.")

@router.post("/product/sync", summary="Tạo/Cập nhật sản phẩm (MySQL + Vector DB)")
async def sync_product(payload: ProductUpdatePayload):
    try:
        # BƯỚC 1: Lưu vào cơ sở dữ liệu chính (MySQL)
        upsert_product_to_mysql(payload)
        
        # BƯỚC 2: Đồng bộ sang Vector DB (Qdrant) cho hệ thống AI
        sync_product_to_vector_db(
            product_id=payload.product_id,
            name=payload.name,
            description=payload.description,
            category=payload.category,
            price=payload.price
        )
        
        return {
            "status": "success", 
            "message": f"Sản phẩm {payload.product_id} đã được lưu vào MySQL và đồng bộ lên Qdrant!"
        }
    
    except Exception as e:
        logger.error(f"Lỗi API Sync: {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý dữ liệu. Vui lòng kiểm tra log.")