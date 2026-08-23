import sys
import gc
import torch
import os
import random
import warnings  
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import transformers
from torch.optim import AdamW  

# 系統硬體調優
available_cpus = os.cpu_count() or 4
torch.set_num_threads(max(1, available_cpus - 2))
torch.set_num_interop_threads(max(1, available_cpus - 2))

# 1. 停用引發硬體資源超限的 Flex Attention 編譯
os.environ["TORCH_DISABLE_FENTION"] = "1"
# 2. 透過環境變數直接關閉 PyTorch Inductor 的後端編譯
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["HF_TOKEN"] = ""

if torch.cuda.is_available():
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 3405

def force_clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def create_model_and_tokenizer():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ [CUDA 錯誤] 當前系統未偵測到可用的 NVIDIA GPU！此任務已限定僅能使用 GPU 執行。")

    model_name = "unsloth/medgemma-1.5-4b-it"
    print(f"🚀 [Task - GPU 專用模式] 正在透過 Unsloth 載入模型 {model_name}...")
    import unsloth
    from unsloth import FastVisionModel
    
    force_clear_gpu_cache()
    
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        attn_implementation="sdpa", 
        device_map="cuda:0",
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, processor

# task.py
import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

def prepare_vqa_dataset(csv_path, image_root, num, is_train=True, current_seed=None, seed=42):
    if not os.path.exists(csv_path):
        print(f"⚠️ 找不到 CSV 檔案: {csv_path}，返回空清單。")
        return []

    df = pd.read_csv(csv_path)

    possible_img_cols = [
        'Image.Index', 'Image Index', 'Image_Index', 'image_id', 
        'filename', 'file_name', 'image', 'Image', 'path', 'img'
    ]
    img_col = None
    for col in possible_img_cols:
        if col in df.columns:
            img_col = col
            break

    if img_col is None:
        raise ValueError(f"❌ [CSV 欄位錯誤] 在 {csv_path} 中找不到任何圖片欄位！現有欄位為: {df.columns.tolist()}")

    target_labels = [
     "No_Finding",         
     "Atelectasis", "Cardiomegaly", "Consolidation",     
     "Edema", "Effusion", "Emphysema",              
     "Fibrosis", "Hernia", "Infiltration",              
     "Mass", "Nodule", "Pleural_Thickening",      
     "Pneumonia", "Pneumothorax" , "Tuberculosis"   
    ]

    total_requested = min(num, len(df))
    
    df_healthy = df[df['No_Finding'] == 1]
    df_abnormal = df[df['No_Finding'] == 0]
    
    # 決定當前抽樣使用的 Seed
    sampling_seed = seed if (not is_train or current_seed is None) else current_seed

    local_random = random.Random(sampling_seed)
    abnormal_ratio = local_random.uniform(0.65, 0.7)
    
    n_healthy_target = min(int(total_requested * (1 - abnormal_ratio)), len(df_healthy))
    n_abnormal_target = min(total_requested - n_healthy_target, len(df_abnormal))
    n_healthy_target = min(total_requested - n_abnormal_target, len(df_healthy))

    if not is_train:
        # 🧪 測試/驗證集：固定使用 seed 抽樣
        print(f"\n🧪 [測試/驗證集] 按比例抽取 (固定 Seed: {seed}): 健康 {n_healthy_target} 筆, 異常 {n_abnormal_target} 筆")
        final_healthy = df_healthy.sample(n=n_healthy_target, random_state=seed) if n_healthy_target > 0 else pd.DataFrame()
        final_abnormal = df_abnormal.sample(n=n_abnormal_target, random_state=seed) if n_abnormal_target > 0 else pd.DataFrame()
        df_balanced = pd.concat([final_healthy, final_abnormal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        # 🏋️ 訓練集：使用稀有疾病優先的多疾病與單一疾病分層抽樣
        print(f"\n🏋️ [訓練集 - 動態抽樣] 使用 Seed: {sampling_seed}")
        guaranteed_abnormal_indices = set()
        disease_labels = [l for l in target_labels if l in df.columns and l != "No_Finding"]

        df_abnormal = df_abnormal.copy()
        df_abnormal['disease_count'] = df_abnormal[disease_labels].sum(axis=1)

        # 1. 區分為多疾病群 (疾病數 > 1) 與 單一疾病群 (疾病數 = 1)
        df_multi = df_abnormal[df_abnormal['disease_count'] > 1]
        df_single = df_abnormal[df_abnormal['disease_count'] == 1]

        # 2. 多疾病群：動態尋找「當前最少量的疾病標籤」並優先抽取
        print("🌊 [多疾病群] 開始依疾病稀有度由少到多動態提取...")
        while True:
            still_needed = n_abnormal_target - len(guaranteed_abnormal_indices)
            if still_needed <= 0:
                break

            # 取得多疾病群中尚未被抽取的剩餘池
            df_multi_remaining = df_multi[~df_multi.index.isin(guaranteed_abnormal_indices)]
            if df_multi_remaining.empty:
                print("  └ 多疾病群已全部抽取完畢。")
                break

            # 統計剩餘多疾病池中，各疾病標籤出現的陽性總數
            class_counts = {l: (df_multi_remaining[l] == 1).sum() for l in disease_labels}
            class_counts = {k: v for k, v in class_counts.items() if v > 0}

            if not class_counts:
                break

            # 找出出現次數最少（最稀有）的疾病標籤
            min_label = min(class_counts, key=class_counts.get)
            df_min_label_samples = df_multi_remaining[df_multi_remaining[min_label] == 1]
            available_count = len(df_min_label_samples)

            if available_count <= still_needed:
                # 該稀有疾病的多疾病樣本數 <= 剩餘名額 -> 全抽
                guaranteed_abnormal_indices.update(df_min_label_samples.index.tolist())
                print(f"  └ 稀有標籤 [{min_label}] (剩餘 {available_count} 筆) -> 全抽，累積異常: {len(guaranteed_abnormal_indices)}/{n_abnormal_target}")
            else:
                # 該稀有疾病的多疾病樣本數 > 剩餘名額 -> 隨機抽樣補滿並結束
                sampled = df_min_label_samples.sample(n=still_needed, random_state=sampling_seed)
                guaranteed_abnormal_indices.update(sampled.index.tolist())
                print(f"  └ 稀有標籤 [{min_label}] (剩餘 {available_count} 筆) -> 隨機抽取 {still_needed} 筆補滿，達目標額度")
                break

        # 3. 若多疾病群抽完仍未達到目標數量，從單一疾病群隨機補足
        still_needed = n_abnormal_target - len(guaranteed_abnormal_indices)
        if still_needed > 0 and not df_single.empty:
            df_single_remaining = df_single[~df_single.index.isin(guaranteed_abnormal_indices)]
            if not df_single_remaining.empty:
                draw_num = min(still_needed, len(df_single_remaining))
                sampled_single = df_single_remaining.sample(n=draw_num, random_state=sampling_seed)
                guaranteed_abnormal_indices.update(sampled_single.index.tolist())
                print(f"🍂 [單一疾病群] 多疾病群不足，隨機補充 {draw_num} / {len(df_single_remaining)} 筆")

        # 4. 備用機制：若極端情況下仍有名額缺口，從剩餘所有異常樣本中補齊
        still_needed = n_abnormal_target - len(guaranteed_abnormal_indices)
        if still_needed > 0:
            df_remaining = df_abnormal[~df_abnormal.index.isin(guaranteed_abnormal_indices)]
            if not df_remaining.empty:
                sampled_backup = df_remaining.sample(n=min(still_needed, len(df_remaining)), random_state=sampling_seed)
                guaranteed_abnormal_indices.update(sampled_backup.index.tolist())

        final_abnormal = df_abnormal.loc[list(guaranteed_abnormal_indices)]

        # 5. 抽取健康組並打亂
        if n_healthy_target > 0 and len(df_healthy) > 0:
            final_healthy = df_healthy.sample(n=n_healthy_target, random_state=sampling_seed)
            df_balanced = pd.concat([final_abnormal, final_healthy]).sample(frac=1, random_state=sampling_seed).reset_index(drop=True)
        else:
            df_balanced = final_abnormal.sample(frac=1, random_state=sampling_seed).reset_index(drop=True)

    print("\n📊 —— 當前站點 Local 資料集各標籤分佈陽性個數 ——")
    for label in target_labels:
        pos_count = (df_balanced[label] == 1).sum() if label in df_balanced.columns else 0
        print(f"  * {label.ljust(20)}: {pos_count} 筆")
    print(f"總計總樣本數: {len(df_balanced)} 筆")
    print("——————————————————————————————————————————————\n")

    if len(df_balanced) == 0:
        return []

    converted_data = []
    missing_count = 0
    first_failed_path = None

    for _, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc="Processing Dataset"):
        instruction = "Review this chest radiograph and provide the clinical impressions based on the visible findings."
        
        raw_label = "No Finding"
        if 'answer' in row:
            raw_label = str(row['answer']).strip()
        elif 'Finding.Labels' in row:
            raw_label = str(row['Finding.Labels']).strip()

        if raw_label in ["No Finding", "nan", ""] or ('No_Finding' in row and row['No_Finding'] == 1):
            answer = "No abnormalities"
        else:
            labels = [l.strip() for l in raw_label.split('|') if l.strip()]
            labels_clean = [l.lower() for l in labels]
            
            if len(labels_clean) == 1:
                answer = f"{labels_clean[0]}"
            else:
                findings_str = ", ".join(labels_clean[:-1]) + f" and {labels_clean[-1]}"
                answer = f"{findings_str}"

        pic_val = row[img_col]
        pic_str = str(pic_val).strip().lstrip('/')

        if not pic_str or pic_str == 'nan':
            missing_count += 1
            continue

        possible_paths = [
            os.path.join(image_root, pic_str),
            os.path.join(image_root, os.path.basename(pic_str)),
            os.path.join(os.path.dirname(image_root), pic_str),
            os.path.join(os.path.dirname(image_root), os.path.basename(pic_str))
        ]

        valid_img_path = None
        for p in possible_paths:
            if os.path.exists(p):
                valid_img_path = p
                break

        if valid_img_path is None:
            missing_count += 1
            if first_failed_path is None:
                first_failed_path = possible_paths[0]
            continue

        try:
            image = Image.open(valid_img_path).convert("RGB")
            image.load()
        except Exception:
            missing_count += 1
            continue

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}, {"type": "image"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        converted_data.append({
            "messages": messages,
            "images": [image]
        })

    if missing_count > 0:
        print(f"⚠️ [警告] 共有 {missing_count} 筆影像檔案不存在或無法載入，已自動忽略。")
        if first_failed_path:
            print(f"🔍 [除錯訊息] 嘗試讀取的首筆無效路徑為: {first_failed_path}")

    print(f"✅ 成功載入 {len(converted_data)} 筆有效 Vision-VQA 樣本。")
    return converted_data

def train_local(model, processor, train_dataset, epochs=1):
    if not train_dataset:
        print("⚠️ [Task] 本地訓練集為空，跳過此回合。")
        return 0.0, {"loss": 0.0}

    if not torch.cuda.is_available():
        raise RuntimeError("❌ [CUDA 錯誤] 當前環境無可用 GPU，無法執行 train_local。")

    force_clear_gpu_cache()

    print("🚀 使用 Unsloth 進行高效 GPU 微調...")
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig
    from unsloth import is_bf16_supported

    model.to("cuda")
    model.train()
    
    collator = UnslothVisionDataCollator(model, processor)

    training_args = SFTConfig(
        per_device_train_batch_size=1,     
        gradient_accumulation_steps=4,     
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",                                
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        dataset_text_field="text",
        max_seq_length=1024,
        dataset_num_proc=1,
        packing=False,
    )

    try:
        from unsloth import SFTTrainer as UnslothSFTTrainer
    except ImportError:
        from trl import SFTTrainer as UnslothSFTTrainer

    trainer = UnslothSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
        args=training_args,
    )

    print("🏋️ [Task] 開始本地微調週期...")
    train_result = trainer.train()
    metrics = train_result.metrics
    avg_loss = metrics.get("train_loss", 0.0)
    
    del trainer
    force_clear_gpu_cache()
    return avg_loss, metrics

def evaluate_local(model, processor, test_dataset):
    if not test_dataset:
        print("⚠️ [Task] 本地測試集為空，跳過評估。")
        return 0.0, {"eval_loss": 0.0}

    if not torch.cuda.is_available():
        raise RuntimeError("❌ [CUDA 錯誤] 當前環境無可用 GPU，無法執行 evaluate_local。")

    force_clear_gpu_cache()

    device = torch.device("cuda")
    from unsloth.trainer import UnslothVisionDataCollator
    collator = UnslothVisionDataCollator(model, processor)

    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collator, num_workers=0)
    total_loss = 0.0
    
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model.to(device=device, dtype=target_dtype)
    model.eval()
    
    print(f"🚀 [Task] GPU 模型評估中 (精度: {target_dtype})...")
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {
                k: v.to(device=device, dtype=target_dtype) 
                if (isinstance(v, torch.Tensor) and torch.is_floating_point(v))
                else v.to(device) if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            outputs = model(**batch)
            total_loss += outputs.loss.item()
            
    avg_loss = total_loss / max(len(eval_dataloader), 1)
    
    del eval_dataloader
    force_clear_gpu_cache()
    return avg_loss, {"eval_loss": avg_loss}

def get_weights(model):
    if model is None or not hasattr(model, "state_dict"):
        return []
    return [
        val.detach().to(torch.float32).cpu().numpy() 
        for name, val in model.state_dict().items() 
        if "lora_" in name
    ]

def set_weights(model, parameters):
    if model is None or not hasattr(model, "state_dict") or not parameters:
        return
    state_dict = model.state_dict()
    lora_keys = [k for k in state_dict.keys() if "lora_" in k]
    
    if len(parameters) != len(lora_keys):
        print(f"⚠️ [警告] 伺服器參數量 ({len(parameters)}) 與本地 LoRA 數量 ({len(lora_keys)}) 不符！跳過同步。")
        return

    for key, updated_ndarray in zip(lora_keys, parameters):
        state_dict[key] = torch.from_numpy(updated_ndarray).to(state_dict[key].device)
    
    model.load_state_dict(state_dict, strict=False)
    print("🔄 [Task] 成功覆寫本地 LoRA 權重結構。")
