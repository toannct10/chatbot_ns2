import json
import uuid
import os
# ✅ Đổi: bỏ openai, dùng google-generativeai
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# 1. Khởi tạo Client
# ✅ Đổi: dùng Gemini embedding thay vì OpenAI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# Kết nối tới Qdrant (thực tế sẽ là URL của Qdrant server)
qdrant = QdrantClient(url="http://qdrant_db:6333")
COLLECTION_NAME = "ecommerce_products"

def setup_qdrant():
    """Tạo collection và đánh index cho metadata"""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            # ✅ Đổi: Gemini text-embedding-004 có chiều 768 (OpenAI là 1536)
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"✅ Đã tạo collection: {COLLECTION_NAME}")
        
        # Đánh index cho database giống hệt tư duy làm việc với MySQL/PostgreSQL
        qdrant.create_payload_index(COLLECTION_NAME, "price", PayloadSchemaType.FLOAT)
        qdrant.create_payload_index(COLLECTION_NAME, "category", PayloadSchemaType.KEYWORD)
        print("✅ Đã tạo Payload Index cho price và category.")

def embed_and_load(input_file, batch_size=100):
    print("⏳ Đang tiến hành Vector hóa và Load vào Qdrant...")
    points = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk_data = json.loads(line)
            
            # ✅ Đổi: gọi Gemini Embedding API thay vì OpenAI
            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=chunk_data["content"],
                task_type="retrieval_document",  # tối ưu cho RAG indexing
            )
            vector = response["embedding"]
            
            # Tạo UUID từ chunk_id để đảm bảo tính duy nhất (Upsert)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_data["chunk_id"]))
            
            # Tạo bản ghi (Point) cho Qdrant
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk_data["chunk_id"],
                    "parent_doc_id": chunk_data["parent_doc_id"],
                    "content": chunk_data["content"],
                    **chunk_data["metadata"]
                }
            )
            points.append(point)
            
            # Load theo Batch để tối ưu I/O và Network
            if len(points) >= batch_size:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"🔄 Đã upsert {len(points)} chunks...")
                points = []  # Reset batch
                
        # Upsert những chunks còn sót lại cuối cùng
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"🔄 Đã upsert {len(points)} chunks cuối cùng.")
            
    print("Hoàn thành Data Pipeline! Dữ liệu đã sẵn sàng để truy vấn.")


if __name__ == "__main__":
    import logging

    # Setup logging cơ bản
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_input_file = "/app/data/test_chunked_products.jsonl"
    
    print("🚀 BẮT ĐẦU TEST: Nhúng Vector (Gemini) và nạp vào Qdrant...")
    
    # Kiểm tra xem có file từ bước Transform không
    if not os.path.exists(test_input_file):
        print(f"❌ Lỗi: Không tìm thấy file {test_input_file}. Hãy chạy test transform trước.")
    else:
        try:
            # 1. Khởi tạo Collection và cấu hình Index (chạy 1 lần)
            print("--- Đang setup Collection ---")
            setup_qdrant()
            
            # 2. Đọc file JSONL, gọi Gemini nhúng và đẩy lên Qdrant
            print("\n--- Đang Embed và Load ---")
            embed_and_load(test_input_file, batch_size=2)  # Đặt batch nhỏ để test cho nhanh
            
            # 3. KIỂM TRA THỰC TẾ: Query ngược lại Qdrant xem có data không
            print("\n👀 KIỂM TRA KẾT QUẢ TRONG QDRANT:")
            
            # Lấy thông tin tổng quan của Collection
            collection_info = qdrant.get_collection(COLLECTION_NAME)
            print(f"- Tên Collection: {COLLECTION_NAME}")
            print(f"- Tổng số Vector (Chunks) đang có: {collection_info.points_count}")
            
            # Kéo thử 1 bản ghi ngẫu nhiên ra xem payload
            records, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=1)
            if records:
                print("\n- Dữ liệu (Payload) của một Vector trong Qdrant:")
                print(json.dumps(records[0].payload, indent=2, ensure_ascii=False))
                
        except Exception as e:
            print(f"❌ Test thất bại: {e}")