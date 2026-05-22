from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của khách hàng")
    category: Optional[str] = Field(default=None, description="Lọc theo danh mục (VD: dien_tu, thoi_trang)")
    max_price: Optional[float] = Field(default=None, description="Mức giá tối đa khách muốn tìm")