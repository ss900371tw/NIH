##########HUGGINGFACE####################
library(dplyr)
library(stringr)
library(readr)

# 1. 讀取與處理 NA
df <- read_csv("D:/mimic-cxr/mimic_cxr_metadata.csv") %>%
  mutate(reports_clean = ifelse(is.na(reports), "", reports))

diseases <- list(
  "Atelectasis"        = "Atelectasis",
  "Cardiomegaly"       = "Cardiomegaly",
  "Consolidation"      = "Consolidation",
  "Edema"              = "Edema",
  "Effusion"           = "Effusion",
  "Emphysema"          = "Emphysema",
  "Fibrosis"           = "Fibrosis",
  "Hernia"             = "Hernia",
  "Infiltration"       = "Infiltration",
  "Mass"               = "Mass",
  "Nodule"             = "Nodule",
  "Pleural Thickening" = "Pleural_Thickening",
  "Pneumonia"          = "Pneumonia",
  "Pneumothorax"       = "Pneumothorax",
  "Tuberculosis"       = "Tuberculosis"
)

# 2. 定義常見的否定前綴模式 (Negation Patterns)
# 排除如 "no ", "without ", "free of ", "ruled out " 等否定詞
negation_prefix <- "(?:no|without|free of|rules out|ruled out|denies|negative for)\\s+(?:\\w+\\s+){0,3}"

# 3. 比對病灶：存在關鍵字且「未被否定」才計為 1
for (keyword in names(diseases)) {
  col_name <- diseases[[keyword]]
  
  # 建立判斷條件：
  # (1) 包含該疾病名稱
  # (2) 不符合前幾字帶有否定詞的模式
  pattern_has <- regex(paste0("\\b", keyword, "\\b"), ignore_case = TRUE)
  pattern_neg <- regex(paste0(negation_prefix, keyword, "\\b"), ignore_case = TRUE)
  
  df[[col_name]] <- as.integer(
    str_detect(df$reports_clean, pattern_has) & !str_detect(df$reports_clean, pattern_neg)
  )
}

disease_cols <- as.character(unlist(diseases))

# 4. 計算 No_Finding 欄位
df <- df %>%
  mutate(No_Finding = as.integer(rowSums(select(., all_of(disease_cols))) == 0))

# 5. 高效向量化合成 Finding.Labels (取代耗時的 apply)
# 利用 matrix 運算比對 1 的位置並組合字串
mat <- as.matrix(df[, disease_cols])
labels_vec <- apply(mat, 1, function(row) {
  present <- disease_cols[row == 1]
  if (length(present) == 0) return("No Finding")
  return(paste(present, collapse = "|"))
})

df$Finding.Labels <- labels_vec

# 6. 整理與匯出
target_columns <- c(
  "reports", "file_path", "split", "Finding.Labels",
  "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation", 
  "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
  "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
  "Pneumonia", "Pneumothorax", "Tuberculosis"
)

df_final <- df %>% select(all_of(intersect(target_columns, names(df))))
write_csv(df_final, "D:/mimic-cxr/mimic_cxr_metadata_processed.csv")



##########PHYSIONET####################
# 載入必要套件
# 載入必要套件
library(dplyr)
library(stringr)
library(readr)

# 1. 定義病灶與同義詞
diseases <- list(
  "Atelectasis"        = "atelectasis",
  "Cardiomegaly"       = "(cardiomegaly|enlarged heart)",
  "Consolidation"      = "consolidation",
  "Edema"              = "(edema|pulmonary edema)",
  "Effusion"           = "(effusion|pleural effusion)",
  "Emphysema"          = "emphysema",
  "Fibrosis"           = "fibrosis",
  "Hernia"             = "hernia",
  "Infiltration"       = "infiltration",
  "Mass"               = "mass",
  "Nodule"             = "nodule",
  "Pleural_Thickening" = "(pleural thickening|thickening of the pleura)",
  "Pneumonia"          = "pneumonia",
  "Pneumothorax"       = "pneumothorax",
  "Tuberculosis"       = "(tuberculosis|tb)"
)

# 2. 定義否定語意規則 (Prefix & Suffix Negation)
neg_prefix <- "(?:no|without|free of|rules out|ruled out|denies|negative for|unremarkable for|no evidence of)\\s+(?:\\w+\\s+){0,6}"
neg_suffix <- "\\s+(?:is|was|are)?\\s*(?:ruled out|excluded|negative|absent|unremarkable)"

# 3. 封裝處理函數
process_mimic_df <- function(file_path) {
  # 讀取 CSV
  df <- read.csv(file_path, row.names = 1)
  
  # 安全處理文字拼接：防止 NA 被轉成 "NA" 字串
  df <- df %>%
    mutate(
      text_clean = ifelse(is.na(text), "", text),
      text_aug_clean = ifelse(is.na(text_augment), "", text_augment),
      reports_clean = paste0(text_clean, " ", text_aug_clean)
    )
  
  # 過濾 AP 與 PA 欄位不合規的資料
  df <- df %>% filter(AP != "[]" & PA != "[]")
  
  disease_cols <- names(diseases)
  
  # 執行否定比對
  for (col_name in disease_cols) {
    keyword <- diseases[[col_name]]
    
    pattern_has  <- regex(paste0("\\b", keyword, "\\b"), ignore_case = TRUE)
    pattern_neg1 <- regex(paste0(neg_prefix, keyword, "\\b"), ignore_case = TRUE)
    pattern_neg2 <- regex(paste0("\\b", keyword, neg_suffix), ignore_case = TRUE)
    
    # 有提到，且沒有被前置或後置語意否定，才算 1
    df[[col_name]] <- as.integer(
      str_detect(df$reports_clean, pattern_has) & 
        !str_detect(df$reports_clean, pattern_neg1) &
        !str_detect(df$reports_clean, pattern_neg2)
    )
  }
  
  # 計算 No_Finding
  df <- df %>%
    mutate(No_Finding = as.integer(rowSums(select(., all_of(disease_cols))) == 0))
  
  # 高效向量化合成 Finding.Labels
  mat <- as.matrix(df[, disease_cols])
  df$Finding.Labels <- apply(mat, 1, function(row) {
    present <- disease_cols[row == 1]
    if (length(present) == 0) return("No Finding")
    return(paste(present, collapse = "|"))
  })
  
  # 整理欄位順序
  target_columns <- c(
    "subject_id", "AP", "PA", "text", "text_augment", "Finding.Labels",
    "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation", 
    "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
    "Pneumonia", "Pneumothorax", "Tuberculosis"
  )
  
  df_final <- df %>% select(all_of(intersect(target_columns, names(df))))
  return(df_final)
}

# ---------------------------------------------------------
# 4. 批次執行 Train 與 Validate 資料集處理
# ---------------------------------------------------------

train_path <- "C:/Users/Administrator/Desktop/mimic_cxr_aug_train.csv"
val_path   <- "C:/Users/Administrator/Desktop/mimic_cxr_aug_validate.csv"

# 處理並儲存 Train
df_train <- process_mimic_df(train_path)
write_csv(df_train, train_path)
print("Train 資料集處理完畢！")

# 處理並儲存 Validate
df_val <- process_mimic_df(val_path)
write_csv(df_val, val_path)
print("Validate 資料集處理完畢！")

# 合併並儲存 All
df_all <- rbind(df_train, df_val)
write_csv(df_all, "C:/Users/Administrator/Desktop/mimic_cxr_aug_all.csv")
print("全部資料集已合併並儲存至 mimic_cxr_aug_all.csv！")
