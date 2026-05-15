# V-Miner VLM Pipeline Architecture

Tài liệu này mô tả kiến trúc mới của **V-Miner**, sau khi tích hợp thành công mô hình Layout phân tích sâu từ **Surya OCR** và thay thế backend nhận diện truyền thống bằng **Vision-Language Model (VLM)** qua nền tảng SiliconFlow.

## Sơ đồ luồng xử lý (Mermaid Diagram)

Dưới đây là sơ đồ luồng hoạt động (Workflow) từ khi đọc file PDF cho đến khi xuất ra file Markdown.

```mermaid
flowchart TD
    %% Define Styles
    classDef input fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
    classDef process fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef model fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef external fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    classDef output fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff
    classDef fallback fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff

    A([Input PDF Document]):::input --> B["Tiền xử lý & Trích xuất ảnh"]:::process
    
    B --> C["Phân tích Bố cục / Layout Analysis<br/><i>Surya Layout Wrapper</i>"]:::model
    
    C -- List of Page Images --> D["Surya LayoutPredictor"]:::external
    
    D -- "Layout Result <br/> (Bounding Boxes + Labels)" --> E["Mapper & Filter"]:::process
    E -. "Lọc bỏ Header/Footer (cat_id=2)" .-> E
    E -- Chuyển đổi sang format YOLO --> F{"Phân loại Block <br/>(Category ID)"}:::process
    
    F -- "Text & Title <br/> (cat_id: 0, 1)" --> G["LightOnOCR Text Backend<br/><i>SiliconFlow VLM API</i>"]:::external
    F -- "Table <br/> (cat_id: 5)" --> H["LightOnOCR Table Backend<br/><i>SiliconFlow VLM API</i>"]:::external
    F -- "Hình ảnh & Công thức <br/> (cat_id: 3, 8)" --> I["Các mô hình cục bộ<br/>(RapidOCR / YOLO)"]:::model

    G --> J["Trích xuất cấu trúc<br/><i>OpenAI SDK + Pydantic</i>"]:::process
    H --> K["Trích xuất Bảng HTML<br/><i>Strict Prompting</i>"]:::process
    
    J -- JSON Parsing Success --> L["Văn bản thuần túy"]:::process
    J -- API Error / Markdown text --> M["Regex / Text Fallback"]:::fallback
    M --> L
    
    K --> N["Chuyển đổi HTML sang Markdown<br/><i>Lọc hallucination</i>"]:::process
    
    L --> O["Markdown Assembler <br/>(Gộp luồng)"]:::process
    N --> O
    I --> O
    
    O --> P(["File kết quả: output.md"]):::output
```

## Các thay đổi & Nâng cấp cốt lõi

1. **Surya Layout Analysis (Marker Style):**
   - Thay thế DocLayout-YOLO bằng `surya-ocr` (`LayoutPredictor` + `FoundationPredictor`).
   - Sửa lỗi mapping ID để các label như `Table`, `Title`, `Text` tương thích 100% với luồng pipeline cũ của V-Miner.
   - Thêm bộ lọc (Filter) trực tiếp tại bước Layout để drop các block vô giá trị như `Page-header`, `Page-footer`.

2. **VLM Backend (LightOnOCR qua SiliconFlow):**
   - Thay thế cơ chế gọi API bằng thư viện `requests` thủ công sang **OpenAI Python SDK** chuẩn.
   - Sử dụng tính năng `response_format` của OpenAI SDK kết hợp với Pydantic Schemas để ép mô hình trả về JSON có cấu trúc cứng.
   - Bổ sung hệ thống Fallback 2 lớp: Nếu mô hình từ chối format JSON (trả về raw text hoặc bọc trong thẻ ````json`), hệ thống tự động chạy Regex để extract nội dung.

3. **Chống Ảo giác (Anti-Hallucination) cho Bảng biểu:**
   - Cập nhật System Prompts nghiêm ngặt, ngăn chặn VLM tự ý "thông minh" lặp lại các hậu tố không cần thiết (ví dụ: gộp tên bảng vào từng ô của header).
   - Thiết lập `temperature=0.0` trên toàn bộ các lời gọi API để đảm bảo tính nhất quán (Deterministic).

## Cách kích hoạt Pipeline mới

Cấu hình các biến môi trường sau trước khi chạy lệnh:

```bash
# 1. Cấu hình xác thực API (SiliconFlow / VLM)
export OPENAI_API_BASE="https://api.siliconflow.com/v1"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
export OPENAI_MODEL="Qwen/Qwen3-VL-32B-Instruct"

# 2. Ép sử dụng VLM Backend
export MINERU_TEXT_BACKEND=lighton
export MINERU_TABLE_BACKEND=lighton
export MINERU_IMAGE_BACKEND=lighton

# 3. Ép sử dụng Surya Layout
export MINERU_LAYOUT_BACKEND=surya

# 4. Chạy lệnh
mineru -p input.pdf -o output_dir
```
