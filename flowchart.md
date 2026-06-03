# Sơ đồ luồng hoạt động MinerU (Rebase v3.2)

```mermaid
flowchart TD
    A[User gọi lệnh: mineru -p document.pdf] --> B(Khởi chạy FastAPI Server & Event Loop)
    B --> C{Xác định Backend xử lý}
    C -->|pipeline| D[Pipeline Analyze]
    C -->|vlm| E[VLM Analyze]
    C -->|hybrid| F[Hybrid Auto Engine]
    
    %% Luồng Hybrid (Mặc định cho văn bản)
    F --> G[DocLayout-YOLO Model]
    G --> H{Phân loại vùng hiển thị}
    H -->|Đoạn văn, tiêu đề| I[OCR/Trích xuất text cục bộ]
    H -->|Công thức toán / Bảng| J[Trích xuất Bảng/Toán học nâng cao]
    
    %% Xử lý logic tiếng Việt cho Bảng (Đã thêm ở v3.2)
    J --> K{Bảng hoặc vùng phát hiện Tiếng Việt?}
    K -->|Không có tiếng Việt| M[Gọi Model Table tiêu chuẩn của MinerU]
    K -->|Có tiếng Việt| L[Chuyển hướng cho API / Local MLX]
    
    %% Luồng Local MLX / VLM
    L --> N[Client / LightOnOCR-2-1B]
    N -->|Chạy trên Main Thread (với MLX)| O[GPU Inference]
    
    %% Gom kết quả
    O --> P[Gộp Layout, Text, Table thành Markdown chuẩn]
    M --> P
    I --> P
    P --> Q[Ghi ra thư mục: auto/document.md]
```
