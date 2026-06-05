import pathlib
import os
import sys
import shutil

def find_and_copy_md(source_dir, dest_dir):
    source_path = pathlib.Path(source_dir)
    dest_path = pathlib.Path(dest_dir)
    
    # Tạo thư mục đích nếu chưa tồn tại
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Nguồn: {source_path.absolute()}")
    print(f"Đích:  {dest_path.absolute()}\n")
    
    # Tìm tất cả file .md đệ quy
    md_files = list(source_path.rglob("*.md"))
    
    if not md_files:
        print("Không tìm thấy file Markdown nào.")
        return

    print(f"Tìm thấy {len(md_files)} file(s). Đang tiến hành copy...")
    
    count = 0
    for file in md_files:
        try:
            target_file = dest_path / file.name
            
            # Xử lý trùng tên: nếu file đã tồn tại ở đích, đánh số thứ tự
            if target_file.exists():
                stem = file.stem
                suffix = file.suffix
                counter = 1
                while (dest_path / f"{stem}_{counter}{suffix}").exists():
                    counter += 1
                target_file = dest_path / f"{stem}_{counter}{suffix}"
            
            shutil.copy2(file, target_file)
            print(f"✔ Đã copy: {file.name} -> {target_file.name}")
            count += 1
        except Exception as e:
            print(f"✘ Lỗi khi copy {file.name}: {e}")

    print(f"\nHoàn tất! Đã copy thành công {count}/{len(md_files)} file.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Sử dụng: python find_md.py <thư_mục_nguồn> <thư_mục_đích>")
        sys.exit(1)
        
    src = sys.argv[1]
    dst = sys.argv[2]
    
    find_and_copy_md(src, dst)
