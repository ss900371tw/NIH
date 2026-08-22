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
