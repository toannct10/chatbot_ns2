import os
import uuid
import logging
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
logger = logging.getLogger(__name__)
 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
 
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant_db:6333"))
COLLECTION_NAME = "ecommerce_products"
 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
 
 
def sync_service_to_vector_db(
    service_id: str,
    name: str,
    description: str,
    service_type: str,   # "deepscan_cv" | "mooc_interview" | "pricing" | "faq" | "support"
    content_type: str,   # "service_info" | "pricing" | "faq"
):
    """
    Đồng bộ 1 nội dung dịch vụ NextStep vào Qdrant.
    Áp dụng nguyên tắc: Xóa chunk cũ -> Tạo chunk mới -> Upsert.
    """
    doc_id = f"service_{service_id}"
    logger.info(f"Đang đồng bộ nội dung {doc_id} lên Qdrant...")
 
    try:
        # 1. XÓA CHUNK CŨ
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="parent_doc_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                    ],
                )
            ),
        )
        logger.info(f"Đã dọn dẹp các chunk cũ của {doc_id}")
 
        safe_desc = description if description else "Không có mô tả chi tiết."
        content_text = f"Tên: {name}\nNội dung: {safe_desc}"
 
        chunks = text_splitter.split_text(content_text)
        points = []
 
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            enriched_text = f"[Thuộc dịch vụ: {name}] {chunk_text}"
 
            embed_res = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=enriched_text,
                task_type="retrieval_document",
            )
            vector = embed_res["embedding"]
 
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk_id,
                        "parent_doc_id": doc_id,
                        "content": enriched_text,
                        "service_type": service_type,
                        "type": content_type,
                    }
                )
            )
 
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Đã Upsert thành công {len(points)} chunks mới cho {doc_id}.")
 
    except Exception as e:
        logger.error(f"Lỗi khi đồng bộ {doc_id}: {str(e)}")
        raise
 
 
# ── Seed dữ liệu mẫu từ FAQ PDF ──────────────────────────────────────────────
 
NEXTSTEP_KNOWLEDGE_BASE = [
    {
        "id": "intro",
        "name": "Giới thiệu NextStep",
        "service_type": "general",
        "content_type": "service_info",
        "description": """NextStep là nền tảng hỗ trợ người tìm việc cải thiện hồ sơ và kỹ năng phỏng vấn thông qua trí tuệ nhân tạo (AI).
Website cung cấp hai tính năng chính:
- DeepScan CV: Chấm điểm và đánh giá CV dựa trên mô tả công việc (Job Description).
- Mooc Interview AI: Mô phỏng buổi phỏng vấn 1-1 với chuyên gia HR AI.
Mục tiêu: giúp người dùng tối ưu CV, luyện tập phỏng vấn thực tế và nhận phản hồi chuyên nghiệp.""",
    },
    {
        "id": "deepscan_cv",
        "name": "DeepScan CV",
        "service_type": "deepscan_cv",
        "content_type": "service_info",
        "description": """DeepScan CV cho phép người dùng upload CV và nhập Job Description của vị trí muốn ứng tuyển.
Hệ thống AI sẽ phân tích CV, so sánh với yêu cầu công việc và chấm điểm mức độ phù hợp.
Sau khi phân tích, hệ thống cung cấp:
- Điểm đánh giá CV
- Nhận xét về nội dung CV
- Những điểm mạnh của CV
- Những điểm cần cải thiện
- Gợi ý chỉnh sửa để CV phù hợp hơn với vị trí ứng tuyển""",
    },
    {
        "id": "mooc_interview",
        "name": "Mooc Interview AI",
        "service_type": "mooc_interview",
        "content_type": "service_info",
        "description": """Mooc Interview AI là hệ thống phỏng vấn mô phỏng với AI đóng vai trò chuyên gia HR.
Tính năng bao gồm:
- Phỏng vấn 1-1 với AI
- AI đưa ra câu hỏi phù hợp với ngành nghề, vị trí và kinh nghiệm ứng viên
- Mô phỏng quy trình phỏng vấn thực tế
Sau buổi phỏng vấn, hệ thống sẽ đánh giá câu trả lời, phân tích điểm mạnh/yếu và gợi ý cải thiện kỹ năng.""",
    },
    {
        "id": "pricing_free",
        "name": "Gói Free",
        "service_type": "pricing",
        "content_type": "pricing",
        "description": """Gói Free - Miễn phí hoàn toàn, phù hợp để trải nghiệm thử hệ thống.
Giới hạn sử dụng:
- DeepScan CV: 1 lần / 48 giờ
- Gợi ý chỉnh sửa CV
- 3 câu hỏi phỏng vấn cơ bản""",
    },
    {
        "id": "pricing_standard",
        "name": "Gói Standard",
        "service_type": "pricing",
        "content_type": "pricing",
        "description": """Gói Standard - Mua lẻ theo từng lần sử dụng.
Chi phí: 15.000 VNĐ / lần
Bao gồm:
- 1 lần DeepScan CV
- 1 lần Mooc Interview (15 phút)
- Đánh giá và gợi ý cải thiện CV
Phù hợp cho người dùng muốn sử dụng từng lần khi cần.""",
    },
    {
        "id": "pricing_premium",
        "name": "Gói Premium",
        "service_type": "pricing",
        "content_type": "pricing",
        "description": """Gói Premium - Dành cho người dùng luyện phỏng vấn thường xuyên hoặc tối ưu CV nhiều lần.
Quyền lợi: Tối đa 20 lần mỗi ngày cho các tính năng:
- Mooc Interview
- DeepScan CV
- Feedback CV""",
    },
    {
        "id": "payment",
        "name": "Thanh toán",
        "service_type": "payment",
        "content_type": "service_info",
        "description": """NextStep hỗ trợ các phương thức thanh toán: VNPAY và MoMo.
Sau khi thanh toán thành công, hệ thống sẽ tự động kích hoạt quyền sử dụng tương ứng với gói đã mua.""",
    },
    {
        "id": "support",
        "name": "Hỗ trợ khách hàng",
        "service_type": "support",
        "content_type": "service_info",
        "description": """Nếu gặp các vấn đề như lỗi hệ thống, không sử dụng được tính năng, lỗi thanh toán hoặc cần hỗ trợ tài khoản:
Email hỗ trợ: ngcongtoan10@gmail.com
Chúng tôi sẽ phản hồi trong thời gian sớm nhất.""",
    },
    {
        "id": "faq",
        "name": "Câu hỏi thường gặp",
        "service_type": "general",
        "content_type": "faq",
        "description": """Q: NextStep dùng để làm gì?
A: NextStep giúp người dùng cải thiện CV và luyện tập phỏng vấn thông qua AI.
 
Q: DeepScan CV hoạt động như thế nào?
A: Bạn tải CV lên và nhập mô tả công việc. AI sẽ phân tích CV, chấm điểm và đưa ra gợi ý cải thiện.
 
Q: Mooc Interview AI là gì?
A: Hệ thống phỏng vấn mô phỏng với AI đóng vai chuyên gia HR, đặt câu hỏi và đánh giá câu trả lời.
 
Q: Gói Free có miễn phí hoàn toàn không?
A: Có, nhưng có giới hạn số lần sử dụng.
 
Q: Tôi có thể thanh toán bằng cách nào?
A: NextStep hỗ trợ thanh toán qua VNPAY và MoMo.
 
Q: Nếu gặp lỗi thì liên hệ ở đâu?
A: Gửi email tới ngcongtoan10@gmail.com""",
    },
]
 
 
def seed_knowledge_base():
    """Seed toàn bộ dữ liệu NextStep vào Qdrant lần đầu."""
    logger.info("Bắt đầu seed dữ liệu NextStep vào Qdrant...")
    for item in NEXTSTEP_KNOWLEDGE_BASE:
        sync_service_to_vector_db(
            service_id=item["id"],
            name=item["name"],
            description=item["description"],
            service_type=item["service_type"],
            content_type=item["content_type"],
        )
    logger.info("✅ Seed dữ liệu hoàn tất!")