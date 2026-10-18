"""
Helper module: Tìm kiếm và giải quyết đường dẫn tệp dữ liệu thông minh cho FoodHealthQA.
Xử lý linh hoạt việc Windows ẩn đuôi tệp (.csv, .xlsx, .json) hoặc đặt tên tệp khác nhau.
"""
import os
import pandas as pd

def find_data_file(data_dir, base_names, extensions=(".csv", ".xlsx", ".json", "")):
    """
    Tìm kiếm thông minh tệp dữ liệu trong thư mục data_dir theo danh sách tên cơ sở và đuôi tệp.
    Hỗ trợ xử lý trường hợp Windows ẩn đuôi tệp hoặc người dùng dùng JSON/CSV/XLSX.
    """
    if isinstance(base_names, str):
        base_names = [base_names]
    
    # 1. Thử trực tiếp nếu bname đã là đường dẫn hoàn chỉnh
    for bname in base_names:
        direct = os.path.join(data_dir, bname) if not os.path.isabs(bname) else bname
        if os.path.exists(direct) and os.path.isfile(direct):
            return direct
            
    # 2. Thử ghép base_name với các đuôi mở rộng
    for bname in base_names:
        clean_name = os.path.splitext(bname)[0]
        for ext in extensions:
            cand = os.path.join(data_dir, f"{clean_name}{ext}")
            if os.path.exists(cand) and os.path.isfile(cand):
                return cand
            cand_raw = os.path.join(data_dir, f"{bname}{ext}")
            if os.path.exists(cand_raw) and os.path.isfile(cand_raw):
                return cand_raw

    return None

def read_table_flexibly(file_path):
    """
    Đọc tệp bảng dữ liệu linh hoạt: hỗ trợ cả CSV (đa encoding), Excel (.xlsx/.xls)
    hoặc tệp không có đuôi mở rộng trên Windows.
    """
    if not file_path or not os.path.exists(file_path):
        return pd.DataFrame()
    
    # Thử đọc CSV với UTF-8
    try:
        return pd.read_csv(file_path, encoding="utf-8").fillna("")
    except Exception:
        pass
        
    # Thử đọc CSV với UTF-8-sig (BOM từ Excel Windows)
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig").fillna("")
    except Exception:
        pass

    # Thử đọc Excel nếu file thực chất là định dạng Excel
    try:
        return pd.read_excel(file_path).fillna("")
    except Exception:
        pass

    # Thử đọc CSV với latin1 / cp1252
    try:
        return pd.read_csv(file_path, encoding="latin1").fillna("")
    except Exception:
        pass

    return pd.DataFrame()
