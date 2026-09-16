import sys
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "kegiatan.md")

def log_activity(category: str, description: str):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        return
        
    new_entry = f"| **{date_str}** | **{time_str}** | **{category}** | {description} |\n"
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    table_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| :--- | :--- | :--- | :--- |"):
            table_index = i
            break
            
    if table_index != -1:
        lines.insert(table_index + 1, new_entry)
    else:
        lines.append(new_entry)
        
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"[OK] Activity logged successfully [{date_str} {time_str}]: [{category}] {description}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        cat = sys.argv[1]
        desc = " ".join(sys.argv[2:])
        log_activity(cat, desc)
    elif len(sys.argv) == 2:
        log_activity("Umum", sys.argv[1])
    else:
        print("Usage: python log_activity.py <Kategori> <Deskripsi Kegiatan>")
        print('Example: python log_activity.py "Evaluation" "Menguji akurasi model pada test dataset"')
