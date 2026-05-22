import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def transform_documents(input_file, output_file):
    print(f"Đang xử lý Transform cho file: {input_file}...")
    # chunk_size=800
    # chunk_overlap=150: Giữ lại 150 ký tự gối đầu giữa 2 chunk để không đứt gãy ngữ cảnh
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    processed_chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            raw_doc = json.loads(line)
            doc_id = raw_doc["doc_id"]
            content = raw_doc["content"]
            metadata = raw_doc["metadata"]
    
            chunks = text_splitter.split_text(content)
            
            for i, chunk_text in enumerate(chunks):
                if metadata.get("type") == "product_info":
                    enriched_text = f"[Thuộc sản phẩm ID: {doc_id}] {chunk_text}"
                else:
                    enriched_text = chunk_text
                
                chunk_record = {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "parent_doc_id": doc_id,
                    "content": enriched_text,
                    "metadata": metadata # inherit metadata (category, price, v.v.)
                }
                processed_chunks.append(chunk_record)

    # Ghi kết quả ra file JSONL mới
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for chunk in processed_chunks:
            out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print(f"✅ Đã Transform xong! Từ file gốc tạo ra được {len(processed_chunks)} chunks.")

if __name__ == "__main__":
    import os
    import logging
    
    # Setup logging cơ bản
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Định nghĩa file đầu vào (lấy luôn kết quả từ bài test Extract) và đầu ra
    test_input_file = "/app/data/test_raw_products.jsonl"
    test_output_file = "/app/data/test_chunked_products.jsonl"
    
    print("🚀 BẮT ĐẦU TEST: Transform (Chia chunk) dữ liệu bằng LangChain...")
    
    if not os.path.exists(test_input_file):
        print(f"❌ Lỗi: Không tìm thấy file input {test_input_file}. Bạn nhớ chạy test extract trước nhé!")
    else:
        try:
            # Chạy hàm transform
            transform_documents(test_input_file, test_output_file)
            
            print("\n👀 KIỂM TRA KẾT QUẢ (In thử 2 chunk đầu tiên):")
            if os.path.exists(test_output_file):
                with open(test_output_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i < 2: # Chỉ in 2 chunk đầu tiên để kiểm tra
                            parsed_json = json.loads(line)
                            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            else:
                print("❌ Lỗi: Không tìm thấy file output.")
                
        except Exception as e:
            print(f"❌ Test thất bại: {e}")