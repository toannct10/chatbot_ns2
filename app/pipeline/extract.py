import json
import os
import logging
from sqlalchemy import create_engine, text
from pypdf import PdfReader

# Cấu hình logging
logger = logging.getLogger(__name__)
DATA_DIR = "/app/data"
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

def extract_pdf_to_jsonl(pdf_path, output_file, doc_category="chinh_sach"):
    print(f"Đang phân tích PDF: {pdf_path}")

    reader = PdfReader(pdf_path)
    extracted_docs = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        # chunk đơn giản
        chunks = split_text(text, max_length=1500)

        for i, chunk in enumerate(chunks):
            doc = {
                "doc_id": f"pdf_{page_num}_{i}",
                "content": chunk,
                "metadata": {
                    "source": pdf_path,
                    "category": doc_category,
                    "type": "pdf_text",
                    "page_number": page_num
                }
            }
            extracted_docs.append(doc)

    # write to jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in extracted_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"✅ Đã trích xuất xong {len(extracted_docs)} đoạn từ PDF vào {output_file}")

def extract_products_to_jsonl(output_file: str):
    """
    Trích xuất toàn bộ sản phẩm 'active' từ MySQL và xuất ra định dạng JSONL.
    Tối ưu hóa bộ nhớ bằng cách đọc theo từng batch (yield_per).
    """
    logger.info(f"Đang kết nối MySQL tại để trích xuất dữ liệu...")

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT product_id, name, description, category, price 
                FROM products 
                WHERE status = 'active'
            """)
            
            # Thực thi query với execution_options(yield_per=1000)
            result = conn.execution_options(yield_per=1000).execute(query)
            
            count = 0
            
            # Mở file để ghi theo dạng append (hoặc ghi đè)
            with open(output_file, 'w', encoding='utf-8') as f:
                for row in result:
                    # 1. Xử lý dữ liệu Null/None từ DB để tránh lỗi Type Error sau này
                    safe_desc = row.description if row.description else "Không có mô tả chi tiết."
                    safe_category = row.category if row.category else "uncategorized"
                    safe_price = float(row.price) if row.price else 0.0
                    
                    # 2. Xây dựng nội dung Text để LLM đọc
                    # Gộp Tên và Mô tả lại thành một khối thống nhất
                    content_text = f"Tên sản phẩm: {row.name}\nMô tả: {safe_desc}"
                    
                    # 3. Đóng gói chuẩn JSON cho Data Pipeline
                    doc = {
                        "doc_id": f"prod_{row.product_id}",
                        "content": content_text,
                        "metadata": {
                            "category": safe_category,
                            "price": safe_price,
                            "type": "product_info"
                        }
                    }
                    
                    # Ghi từng dòng JSON ra file
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    count += 1
                    
                    if count % 5000 == 0:
                        logger.info(f"🔄 Đã trích xuất {count} sản phẩm...")
                        
            logger.info(f"✅ HOÀN TẤT: Đã trích xuất thành công {count} sản phẩm ra file {output_file}.")
            
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất dữ liệu từ MySQL: {str(e)}")
        raise

def split_text(text, max_length=1500):
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end

    return chunks

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    
    test_output_file = "/app/data/test_raw_products.jsonl"
    
    print("BẮT ĐẦU TEST: Trích xuất dữ liệu từ MySQL...")
    try:
        extract_products_to_jsonl(test_output_file)
        
        print("\nKIỂM TRA KẾT QUẢ (In thử 2 dòng đầu tiên):")
        if os.path.exists(test_output_file):
            with open(test_output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 2: # Chỉ in 2 sản phẩm đầu
                        # Format lại JSON cho dễ nhìn trên terminal
                        parsed_json = json.loads(line)
                        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        else:
            print("Lỗi: Không tìm thấy file output.")
            
    except Exception as e:
        print(f"Test thất bại: {e}")
