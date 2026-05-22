import json
import uuid
import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# 1. Khởi tạo Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Kết nối tới Qdrant
qdrant = QdrantClient(url="http://qdrant_db:6333")
COLLECTION_NAME = "ecommerce_products"

def setup_qdrant():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"✅ Đã tạo collection: {COLLECTION_NAME}")
        qdrant.create_payload_index(COLLECTION_NAME, "price", PayloadSchemaType.FLOAT)
        qdrant.create_payload_index(COLLECTION_NAME, "category", PayloadSchemaType.KEYWORD)
        print("✅ Đã tạo Payload Index cho price và category.")

def embed_and_load(input_file, batch_size=100):
    print("⏳ Đang tiến hành Vector hóa và Load vào Qdrant...")
    points = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk_data = json.loads(line)
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=chunk_data["content"],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            vector = response.embeddings[0].values
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_data["chunk_id"]))
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
            if len(points) >= batch_size:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f"🔄 Đã upsert {len(points)} chunks...")
                points = []
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"🔄 Đã upsert {len(points)} chunks cuối cùng.")
    print("✅ Hoàn thành! Dữ liệu đã sẵn sàng để truy vấn.")