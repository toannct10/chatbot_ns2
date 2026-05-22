import os
import time
import logging
from .extract import extract_products_to_jsonl, extract_pdf_to_jsonl
from .transform import transform_documents
from .load import setup_qdrant, embed_and_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_DIR = "/app/data"
RAW_PRODUCTS = os.path.join(DATA_DIR, "raw_products.jsonl")
RAW_POLICIES = os.path.join(DATA_DIR, "raw_policies.jsonl")
CHUNKED_PRODUCTS = os.path.join(DATA_DIR, "chunked_products.jsonl")
CHUNKED_POLICIES = os.path.join(DATA_DIR, "chunked_policies.jsonl")

def run_etl_pipeline():
    start_time = time.time()
    logger.info("BẮT ĐẦU CHẠY DATA PIPELINE (ETL) CHO E-COMMERCE RAG")

    try:
        # ==========================================
        # BƯỚC 1: EXTRACT (Trích xuất)
        # ==========================================
        logger.info("--- BƯỚC 1: EXTRACT ---")
        # 1.1 Trích xuất từ Database MySQL/PostgreSQL
        extract_products_to_jsonl(RAW_PRODUCTS) 
        
        # 1.2 Trích xuất từ file PDF (Ví dụ: file chính sách nằm trong /app/data/policies.pdf)
        pdf_path = os.path.join(DATA_DIR, "chinh_sach_doi_tra.pdf")
        if os.path.exists(pdf_path):
            extract_pdf_to_jsonl(pdf_path, RAW_POLICIES, doc_category="chinh_sach")
        else:
            logger.warning(f"Không tìm thấy file PDF tại {pdf_path}. Bỏ qua...")

        # ==========================================
        # BƯỚC 2: TRANSFORM (Làm sạch & Cắt chunk)
        # ==========================================
        logger.info("--- BƯỚC 2: TRANSFORM ---")
        if os.path.exists(RAW_PRODUCTS):
            transform_documents(RAW_PRODUCTS, CHUNKED_PRODUCTS)
        if os.path.exists(RAW_POLICIES):
            transform_documents(RAW_POLICIES, CHUNKED_POLICIES)

        # ==========================================
        # BƯỚC 3: LOAD (Embed & Đẩy vào Qdrant)
        # ==========================================
        logger.info("--- BƯỚC 3: LOAD (VECTOR DB) ---")
        setup_qdrant()
        
        if os.path.exists(CHUNKED_PRODUCTS):
            embed_and_load(CHUNKED_PRODUCTS, batch_size=200)
        if os.path.exists(CHUNKED_POLICIES):
            embed_and_load(CHUNKED_POLICIES, batch_size=200)

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(f"DATA PIPELINE HOÀN TẤT THÀNH CÔNG SAU {elapsed_time} GIÂY!")

    except Exception as e:
        logger.error(f"PIPELINE THẤT BẠI: {str(e)}")
        # Trong thực tế, bạn có thể bắn alert về Slack/Telegram ở đây

if __name__ == "__main__":
    run_etl_pipeline()