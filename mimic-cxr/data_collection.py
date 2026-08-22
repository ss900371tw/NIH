######### HUGGINGFACE ###########
import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 設定路徑（根據您的圖片顯示之路徑）
DATA_DIR = r"D:\mimic-cxr\data"
OUTPUT_IMG_DIR = r"D:\mimic-cxr\images"
OUTPUT_CSV_PATH = r"D:\mimic-cxr\mimic_cxr_metadata.csv"

def extract_parquet_data(data_dir, output_img_dir, output_csv_path):
    os.makedirs(output_img_dir, exist_ok=True)
    
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    print(f"找到 {len(parquet_files)} 個 Parquet 檔案: {parquet_files}")
    
    all_dfs = []

    for pfile in parquet_files:
        p_path = os.path.join(data_dir, pfile)
        split_name = pfile.split('-')[0]  # train / test / validation
        print(f"\n正在讀取 {pfile} ...")
        
        df = pd.read_parquet(p_path)
        
        # 尋找包含圖片數據的欄位名稱 (通常為 'image')
        img_col = next((c for c in df.columns if 'image' in c.lower()), None)
        
        saved_img_paths = []
        
        if img_col:
            print(f"解包圖片中 (欄位名稱: {img_col}) ...")
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                img_data = row[img_col]
                img_filename = f"{split_name}_{idx:06d}.png"
                img_save_path = os.path.join(output_img_dir, img_filename)
                
                try:
                    # 處理 Hugging Face datasets 的圖片格式 (dict 或 bytes)
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        image = Image.open(io.BytesIO(img_data['bytes']))
                    elif isinstance(img_data, bytes):
                        image = Image.open(io.BytesIO(img_data))
                    elif hasattr(img_data, 'save'):
                        image = img_data
                    else:
                        image = None
                        
                    if image:
                        image.convert('RGB').save(img_save_path)
                        saved_img_paths.append(f"images/{img_filename}")
                    else:
                        saved_img_paths.append(None)
                except Exception as e:
                    print(f"第 {idx} 筆圖片儲存失敗: {e}")
                    saved_img_paths.append(None)
            
            # 移除二進位欄位，改補上解包後的圖片相對路徑
            df = df.drop(columns=[img_col])
            df['file_path'] = saved_img_paths
            
        df['split'] = split_name
        all_dfs.append(df)

    # 合併所有表格並輸出為 CSV
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n全部完成！")
    print(f"CSV 已儲存至: {output_csv_path}")
    print(f"圖片已儲存至: {output_img_dir}")

if __name__ == "__main__":
    extract_parquet_data(DATA_DIR, OUTPUT_IMG_DIR, OUTPUT_CSV_PATH)



########### PHYSIONET ###############

import os
import re
import shutil
import pandas as pd

# ==================== 1. 設定路徑 ====================
csv_path = r"C:\Users\Administrator\Desktop\archive\mimic_cxr_all.csv"
output_csv_path = (
    r"C:\Users\Administrator\Desktop\archive\mimic_cxr_all_updated.csv"
)
target_dir = r"C:\Users\Administrator\Desktop\archive\images"

# 【重點修改這裡】請改為包含 files 資料夾的真實上一級目錄
# 例如：如果您的 files 資料夾在 C:\Users\Administrator\Desktop\archive\files
# 這裡就要設定為 r"C:\Users\Administrator\Desktop\archive"
# 如果在其他磁碟機，請改為對應路徑，例如 r"D:\MIMIC_CXR_DATA"
source_base_dir = r"C:\Users\Administrator\Desktop\archive\official_data_iccv_final"

os.makedirs(target_dir, exist_ok=True)

# ==================== 2. 讀取 CSV 檔案 ====================
print("讀取 CSV 檔中...")
df = pd.read_csv(csv_path)

path_columns = ["image", "AP", "PA"]
path_pattern = re.compile(r"files/[^\'\"]+\.jpg")

# ==================== 3. 複製圖片到 images 資料夾 ====================
print("開始複製圖片...")
success_count = 0
fail_count = 0
copied_files = set()

for col in path_columns:
    if col in df.columns:
        for cell in df[col].dropna():
            matches = path_pattern.findall(str(cell))
            for rel_path in matches:
                if rel_path in copied_files:
                    continue

                # 組合絕對路徑
                src_path = os.path.join(source_base_dir, rel_path)
                filename = os.path.basename(rel_path)
                dest_path = os.path.join(target_dir, filename)

                if os.path.exists(src_path):
                    try:
                        shutil.copy2(src_path, dest_path)
                        success_count += 1
                        copied_files.add(rel_path)
                    except Exception as e:
                        print(f"複製失敗 [{src_path}]: {e}")
                        fail_count += 1
                else:
                    fail_count += 1

print(
    f"圖片複製完成！成功複製: {success_count} 個，失敗/未找到: {fail_count} 個。\n"
)

# ==================== 4. 修改 CSV 欄位的路徑文字 ====================
print("開始更新 CSV 裡的路徑文字...")


def update_path_string(text):
    if pd.isna(text):
        return text
    return re.sub(r"files/[^\'\"]+/([^/\'\"]+\.jpg)", r"images/\1", str(text))


for col in path_columns:
    if col in df.columns:
        df[col] = df[col].apply(update_path_string)

df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"CSV 更新完成！已儲存至: {output_csv_path}")
