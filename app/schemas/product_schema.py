from pydantic import BaseModel, Field

class ProductUpdatePayload(BaseModel):
    product_id: int = Field(..., description="ID gốc của sản phẩm")
    name: str
    description: str
    category: str
    price: float