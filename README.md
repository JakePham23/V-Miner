# MinerU - Phụ Lục: Hướng dẫn sử dụng OCR Tiếng Việt

> **Lưu ý Quan Trọng**: Mặc định MinerU vẫn hoạt động theo các hướng dẫn trong **[README-CN.md](README-CN.md)** (hoặc `README_zh-CN.md`). Tài liệu này chỉ dành riêng cho trường hợp bạn cần **Scan Ảnh / PDF chứa nhiều văn bản Tiếng Việt** để đạt chất lượng tốt nhất.

Phiên bản này của MinerU đã được tích hợp thêm khả năng nhận diện tiếng Việt chuyên sâu thông qua **EasyOCR** (cho văn bản thường) và **LightOnOCR API** (cho bảng biểu phức tạp). 

---

## 🚀 1. Cài đặt thư viện cần thiết

Các thư viện cần thiết đã được thêm vào `pyproject.toml` để thuận tiện cho quá trình đóng gói. Nếu bạn cài đặt từ source hoặc dùng virtual environment, các thư viện này sẽ tự động được tải (đặc biệt là `easyocr` và `python-bidi`). 

Nếu cần cài thủ công, chạy lệnh:
```bash
pip install easyocr
```

---

## 📝 2. Hướng dẫn dòng lệnh (CLI)

Để ép hệ thống sử dụng OCR tiếng Việt (EasyOCR) thay vì OCR mặc định của Paddle, bạn cần dùng cờ `-l vi` (language) và chạy thông qua `pipeline` backend.

**Cách 1: Ép buộc chạy OCR Tiếng Việt (Explicit)**
```bash
mineru -p <input_pdf_hoac_anh> -o <thumuc_luu> -b pipeline -l vi -m ocr
```

**Cách 2: Tự động nhận diện (Auto-Routing)**
Nếu bạn dùng chế độ `auto` mà không truyền cờ ngôn ngữ, hệ thống sẽ tự quét vài trang đầu. Nếu phát hiện thấy nhiều ký tự tiếng Việt có dấu, nó sẽ tự động chuyển sang chế độ `-l vi`.
```bash
mineru -p <input_pdf_hoac_anh> -o <thumuc_luu> -b pipeline -m auto
```
*Ghi chú: Đối với các file PDF thuần text thông thường (không phải ảnh scan), chế độ `auto` sẽ vẫn dùng trình trích xuất text mặc định, rất nhanh và không gọi OCR.*

---

## 📊 3. Hướng dẫn cấu hình API cho Bảng biểu (LightOnOCR)

Nhận diện bảng biểu tiếng Việt trong ảnh khá phức tạp. Chúng tôi cung cấp thêm cấu hình gọi API sang các mô hình VLM (Vision-Language Model) tương thích chuẩn OpenAI (ví dụ: mô hình triển khai trên **LM Studio** hoặc **Ollama**).

Nếu bạn muốn bảng biểu được OCR chính xác, hãy bật một Local Server (như LM Studio đang chạy model `lightonocr`) và cấu hình các biến môi trường sau trước khi chạy lệnh `mineru`:

```bash
# Cấu hình cho macOS / Linux
export LIGHTON_SERVER_URL="http://localhost:1234/v1/chat/completions" # Địa chỉ API
export LIGHTON_MODEL_NAME="lightonocr"                                # Tên model bạn đang host
export LIGHTON_API_TYPE="openai"                                      # Loại API: 'openai' hoặc 'ollama'
```

*Cơ chế Fallback: Nếu MinerU không thể kết nối tới API này (server tắt, lỗi mạng, v.v.), hệ thống sẽ không bị crash mà tự động chuyển (fallback) về dùng mô hình nhận diện bảng mặc định của PaddleOCR.*

---

## ❓ Câu hỏi thường gặp
- **Hỏi**: Tại sao kết quả ra tiếng Việt bị mất dấu?
  - **Đáp**: Hãy chắc chắn bạn đã truyền `-b pipeline -l vi` (để gọi EasyOCR) thay vì chạy backend mặc định (`hybrid-auto-engine` dùng VLM Qwen đôi khi không rành tiếng Việt).
---

## 👥 Đội ngũ thực hiện & Đóng góp

Phiên bản tối ưu hóa MinerU cho Tiếng Việt này được thực hiện bởi sự cộng tác giữa:
- **Người dùng (User):** Định hướng kiến trúc, cung cấp yêu cầu nghiệp vụ và tích hợp các công nghệ OCR chuyên sâu (EasyOCR, LightOnOCR).
- **Gemini CLI (AI Assistant):** Hỗ trợ lập trình, thực thi các thay đổi mã nguồn, tối ưu hóa logic phân loại PDF và điều phối luồng xử lý (pipeline).

Chúng tôi cùng nhau tạo ra một công cụ mạnh mẽ hơn để xử lý tài liệu Tiếng Việt một cách tự động và chính xác.
