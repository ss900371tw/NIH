library(tidyverse)
library(dplyr)
library(stringr)
library(purrr)

# 1. 讀取資料
PADCHEST_chest_x_ray_images_labels_160K_01.02.19 <- read.csv(
  "C:/Users/USER/Downloads/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", 
  row.names = 1
)
PADCHEST_chest_x_ray_images_labels_160K_01.02.19<-PADCHEST_chest_x_ray_images_labels_160K_01.02.19[PADCHEST_chest_x_ray_images_labels_160K_01.02.19$Projection %in% c("AP","AP_horizontal","PA"),]
# 2. 嚴格篩選後的標籤映射列表 (已修正：將 "Normal" 統一命名為 "No_Finding")
strict_label_mapping <- list(
  "No_Finding" = c("normal"),
  "Atelectasis" = c(
    "atelectasis", "atelectasis basal", "laminar atelectasis", 
    "lobar atelectasis", "round atelectasis", "segmental atelectasis", 
    "total atelectasis"
  ),
  "Cardiomegaly" = c("cardiomegaly"),
  "Consolidation" = c("consolidation"),
  "Edema" = c("pulmonary edema"),
  "Effusion" = c(
    "pleural effusion", "loculated pleural effusion", 
    "loculated fissural effusion","hydropneumothorax"
  ),
  "Emphysema" = c("emphysema","bullas"),
  "Fibrosis" = c("pulmonary fibrosis","fibrotic band"),
  "Hernia" = c("hiatal hernia"),
  "Infiltration" = c("infiltrates"),
  "Mass" = c(
    "mass", "pulmonary mass","mediastinal mass","soft tissue mass"
  ),
  "Nodule" = c("nodule", "multiple nodules","granuloma","calcified granumola","calcified granuloma"),
  "Pleural_Thickening" = c(
    "pleural thickening", "apical pleural thickening", 
    "calcified pleural thickening", "pleural plaques", 
    "calcified pleural plaques" ,"fissure thickening" , "minor fissure thickening" , "major fissure thickening"
  ),
  "Pneumonia" = c("pneumonia", "atypical pneumonia"),
  "Pneumothorax" = c("pneumothorax","hydropneumothorax"),
  "Tuberculosis" = c("tuberculosis")
)

# 經嚴格篩選後保留的所有獨立原始標籤向量
retained_labels <- unlist(strict_label_mapping, use.names = FALSE)

# 3. 初始列篩選 (利用正則表達式快速過濾)
pattern <- paste0("\\b(", paste(retained_labels, collapse = "|"), ")\\b")

filtered_data <- PADCHEST_chest_x_ray_images_labels_160K_01.02.19[
  grepl(pattern, PADCHEST_chest_x_ray_images_labels_160K_01.02.19$Labels, ignore.case = TRUE), 
]

# 4. 定義清理單一列 Labels 的函數 (已加入 trimws 清除空白)
clean_labels <- function(label_str, valid_labels) {
  # 萃取引號內的標籤文字
  extracted <- unlist(regmatches(label_str, gregexpr("'(.*?)'|\"(.*?)\"", label_str)))
  extracted <- gsub("['\"]", "", extracted) # 去除引號
  
  # 【關鍵修復】去除前後非必要的空白字元 (例如 ' tuberculosis ' -> 'tuberculosis')
  extracted <- trimws(extracted)
  
  # 僅保留存在於 retained_labels 中的項目
  matched <- intersect(extracted, valid_labels)
  
  if (length(matched) == 0) {
    return("[]")
  } else {
    return(paste0("['", paste(matched, collapse = "', '"), "']"))
  }
}

# 套用清理函數
filtered_data$Labels <- sapply(
  filtered_data$Labels, 
  clean_labels, 
  valid_labels = retained_labels
)

# 5. 建立反向查找表與映射函數
reverse_lookup <- unlist(lapply(names(strict_label_mapping), function(target) {
  setNames(rep(target, length(strict_label_mapping[[target]])), strict_label_mapping[[target]])
}))

map_labels_to_targets <- function(label_str) {
  extracted <- str_extract_all(label_str, "(?<=['\"])[^'\"]+(?=['\"])")[[1]]
  
  # 【關鍵修復】去除前後空白，確保查表 key 能完全一致
  extracted <- str_trim(extracted)
  
  mapped <- reverse_lookup[extracted]
  mapped <- mapped[!is.na(mapped)]
  
  unique_mapped <- unique(mapped)
  
  if (length(unique_mapped) == 0) {
    return("[]")
  } else {
    return(str_c("['", str_c(unique_mapped, collapse = "', '"), "']"))
  }
}

# 套用映射函數並剔除空結果
filtered_data$Labels <- sapply(filtered_data$Labels, map_labels_to_targets, USE.NAMES = FALSE)
filtered_data <- filtered_data[!(filtered_data$Labels %in% '[]'), ]

# 6. 生成 16 個標靶類別的 0/1 (One-Hot Encoding) 欄位
target_classes <- c(
  "No_Finding", "Cardiomegaly", "Atelectasis", "Effusion", "Infiltration",
  "Pleural_Thickening", "Nodule", "Pneumonia", "Consolidation", "Hernia",
  "Emphysema", "Mass", "Fibrosis", "Pneumothorax", "Edema", "Tuberculosis"
)

for (target in target_classes) {
  pattern <- paste0("\\b", target, "\\b")
  filtered_data[[target]] <- as.integer(
    str_detect(filtered_data$Labels, pattern)
  )
}

# 7. 計算最終各類別數量統計
label_counts <- filtered_data %>%
  filter(!is.na(Labels) & Labels != "") %>%
  mutate(Labels_clean = str_replace_all(Labels, "[\\[\\]'\"\\}]", "")) %>%
  separate_rows(Labels_clean, sep = ",\\s*") %>%
  mutate(Labels_clean = str_trim(Labels_clean)) %>%
  filter(Labels_clean != "") %>%
  count(Labels_clean, sort = TRUE)

# 檢視結果
head(label_counts, 20)

filtered_data$Finding.Labels <- filtered_data$Labels %>%
  # 去除中括號 [ ] 與單引號 '
  str_replace_all("[\\[\\]']", "") %>%
  # 將逗號與空白 ", " 替換為管線符號 "|"
  str_replace_all(",\\s*", "|") %>%
  # 確保去除多餘的前後空白
  str_trim()

# 如果轉換後有完全空白的狀況（防呆），填入 No_Finding
filtered_data$Finding.Labels[filtered_data$Finding.Labels == ""] <- "No_Finding"

# ==========================================
# 新增步驟：按照指定順序重新排列欄位
# ==========================================
# 定義你希望排在最前面的欄位順序（已將 Tuberculosis 補入）
# ==========================================
# 新增步驟：將指定欄位依照順序移動到「最尾端」
# ==========================================

# 1. 嚴格定義你指定的尾端欄位順序（一併將 Tuberculosis 補在最後面）
desired_order <- c(
  "Finding.Labels", "No_Finding", "Atelectasis", "Cardiomegaly", 
  "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", 
  "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
  "Pneumonia", "Pneumothorax", "Tuberculosis"
)

# 2. 移動欄位：先選取除了這 17 個欄位以外的所有欄位，再把這 17 個欄位依序接在後面
filtered_data <- filtered_data %>%
  select(-all_of(desired_order), all_of(desired_order))



# ==========================================
# 補充步驟：自 df 撈回包含 "lung metastasis" 或 "lepidic adenocarcinoma" 的資料並歸類為 Mass
# ==========================================
df<-PADCHEST_chest_x_ray_images_labels_160K_01.02.19[!(PADCHEST_chest_x_ray_images_labels_160K_01.02.19$ImageID %in% filtered_data$ImageID),]
# 1. 定義特定目標關鍵字
extra_mass_labels <- c("lung metastasis", "lepidic adenocarcinoma","empyema")
extra_pattern <- paste0("\\b(", paste(extra_mass_labels, collapse = "|"), ")\\b")

# 2. 從剩餘的 df 中過濾出包含這兩個標籤的資料列
extra_mass_df <- df[
  grepl(extra_pattern, df$Labels, ignore.case = TRUE), 
]


# ==========================================
# 補充步驟：自 extra_mass_df 依據報告文字精準對應至 NIH 標籤 (支援 Multi-label)
# ==========================================

if (exists("extra_mass_df") && nrow(extra_mass_df) > 0) {
  
  # 1. 動態建立 16 個標靶類別欄位（預設皆為 0）
  for (target in target_classes) {
    extra_mass_df[[target]] <- 0L
  }
  
  # 2. 依據 Report 文字指派對應的 0/1 標籤
  
  # 大尺寸 -> Mass
  mask_mass <- grepl("gran taman|gran tamaño", extra_mass_df$Report, ignore.case = TRUE)
  extra_mass_df$Mass[mask_mass] <- 1L
  
  # 氣胸 -> Pneumothorax
  mask_pneumo <- grepl("neumotorax", extra_mass_df$Report, ignore.case = TRUE)
  extra_mass_df$Pneumothorax[mask_pneumo] <- 1L
  
  # 膿胸/積液 -> Effusion
  mask_effusion <- grepl("empiem", extra_mass_df$Report, ignore.case = TRUE)
  extra_mass_df$Effusion[mask_effusion] <- 1L
  
  # 支氣管徵/肺泡空間病灶 -> 同時給予 Consolidation 與 Infiltration
  mask_con_inf <- grepl("broncogram aere|espaci alveol", extra_mass_df$Report, ignore.case = TRUE)
  extra_mass_df$Consolidation[mask_con_inf] <- 1L
  extra_mass_df$Infiltration[mask_con_inf] <- 1L
  
  # 其餘無上述任何標記的報告 -> Nodule
  mask_nodule <- !(mask_mass | mask_pneumo | mask_effusion | mask_con_inf)
  extra_mass_df$Nodule[mask_nodule] <- 1L
  
  # 3. 根據指派結果，動態反推回字串格式的 Labels 與 Finding.Labels 欄位
  extra_mass_df <- extra_mass_df %>%
    rowwise() %>%
    mutate(
      # 找出值為 1 的類別名稱
      active_targets = list(target_classes[c_across(all_of(target_classes)) == 1]),
      Finding.Labels = paste(active_targets, collapse = "|"),
      Labels = paste0("['", paste(active_targets, collapse = "', '"), "']")
    ) %>%
    ungroup() %>%
    select(-active_targets)
  
  # 4. 確保欄位順序與 filtered_data 完全一致
  extra_mass_df <- extra_mass_df %>%
    select(-all_of(desired_order), all_of(desired_order))
  
  # 5. 與 filtered_data 進行 Row Bind 合併
  filtered_data <- rbind(filtered_data, extra_mass_df)
}

# 6. 寫出最終合併後的 CSV
write.csv(
  filtered_data, 
  "C:/Users/USER/Downloads/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv/PADCHEST.csv"
)
