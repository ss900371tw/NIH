##########HUGGINGFACE####################
# 載入必要套件
library(dplyr)
library(stringr)
library(readr)

# 1. 讀取原始 CSV 檔案
df <-  read.csv("D:/mimic-cxr/mimic_cxr_metadata.csv")

# 2. 處理報告文字中的 NA 值
df <- df %>%
  mutate(reports_clean = ifelse(is.na(reports), "", reports))

# 3. 定義搜尋關鍵字與對應的目標欄位名稱
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

# 4. 比對各病灶文字是否存在 (不分大小寫 regex ignore_case = TRUE)
for (keyword in names(diseases)) {
  col_name <- diseases[[keyword]]
  df[[col_name]] <- as.integer(str_detect(df$reports_clean, regex(keyword, ignore_case = TRUE)))
}

disease_cols <- as.character(unlist(diseases))

# 5. 計算 No_Finding 欄位 (若 15 項病灶皆為 0 則為 1)
df <- df %>%
  mutate(No_Finding = as.integer(rowSums(select(., all_of(disease_cols))) == 0))

# 6. 產生 Finding.Labels 欄位
df$Finding.Labels <- apply(df, 1, function(row) {
  if (as.numeric(row["No_Finding"]) == 1) {
    return("No Finding")
  } else {
    present_diseases <- disease_cols[sapply(disease_cols, function(col) as.numeric(row[col]) == 1)]
    return(paste(present_diseases, collapse = "|"))
  }
})

# 7. 整理與指定欄位順序
target_columns <- c(
  "reports", "file_path", "split", "Finding.Labels",
  "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation", 
  "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
  "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
  "Pneumonia", "Pneumothorax", "Tuberculosis"
)

df_final <- df %>%
  select(all_of(intersect(target_columns, names(df))))

# 8. 匯出處理後的 CSV 檔案
write_csv(df_final, "D:/mimic-cxr/mimic_cxr_metadata_processed.csv")




##########PHYSIONET####################
# 載入必要套件
library(dplyr)
library(stringr)
library(readr)

# 1. 讀取原始 CSV 檔案
df <-  read.csv("C:/Users/Administrator/Desktop/mimic_cxr_aug_train.csv", row.names=1)

df$text_all<-paste0(df$text,df$text_augment)

# 2. 處理報告文字中的 NA 值
df <- df %>%
  mutate(reports_clean = ifelse(is.na(text_all), "", text_all))

df <- df[df$AP!="[]" & df$PA!="[]",]

# 3. 定義搜尋關鍵字與對應的目標欄位名稱
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

# 4. 比對各病灶文字是否存在 (不分大小寫 regex ignore_case = TRUE)
for (keyword in names(diseases)) {
  col_name <- diseases[[keyword]]
  df[[col_name]] <- as.integer(str_detect(df$reports_clean, regex(keyword, ignore_case = TRUE)))
}

disease_cols <- as.character(unlist(diseases))

# 5. 計算 No_Finding 欄位 (若 15 項病灶皆為 0 則為 1)
df <- df %>%
  mutate(No_Finding = as.integer(rowSums(select(., all_of(disease_cols))) == 0))

# 6. 產生 Finding.Labels 欄位
df$Finding.Labels <- apply(df, 1, function(row) {
  if (as.numeric(row["No_Finding"]) == 1) {
    return("No Finding")
  } else {
    present_diseases <- disease_cols[sapply(disease_cols, function(col) as.numeric(row[col]) == 1)]
    return(paste(present_diseases, collapse = "|"))
  }
})

# 7. 整理與指定欄位順序
target_columns <- c(
  "subject_id","AP","PA","text","text_augment", "Finding.Labels",
  "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation", 
  "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
  "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
  "Pneumonia", "Pneumothorax", "Tuberculosis"
)

df_final <- df %>%
  select(all_of(intersect(target_columns, names(df))))

# 8. 匯出處理後的 CSV 檔案
write_csv(df_final, "C:/Users/Administrator/Desktop/mimic_cxr_aug_train.csv")






# 載入必要套件
library(dplyr)
library(stringr)
library(readr)

# 1. 讀取原始 CSV 檔案
df <-  read.csv("C:/Users/Administrator/Desktop/mimic_cxr_aug_validate.csv", row.names=1)

df$text_all<-paste0(df$text,df$text_augment)

# 2. 處理報告文字中的 NA 值
df <- df %>%
  mutate(reports_clean = ifelse(is.na(text_all), "", text_all))

df <- df[df$AP!="[]" & df$PA!="[]",]

# 3. 定義搜尋關鍵字與對應的目標欄位名稱
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

# 4. 比對各病灶文字是否存在 (不分大小寫 regex ignore_case = TRUE)
for (keyword in names(diseases)) {
  col_name <- diseases[[keyword]]
  df[[col_name]] <- as.integer(str_detect(df$reports_clean, regex(keyword, ignore_case = TRUE)))
}

disease_cols <- as.character(unlist(diseases))

# 5. 計算 No_Finding 欄位 (若 15 項病灶皆為 0 則為 1)
df <- df %>%
  mutate(No_Finding = as.integer(rowSums(select(., all_of(disease_cols))) == 0))

# 6. 產生 Finding.Labels 欄位
df$Finding.Labels <- apply(df, 1, function(row) {
  if (as.numeric(row["No_Finding"]) == 1) {
    return("No Finding")
  } else {
    present_diseases <- disease_cols[sapply(disease_cols, function(col) as.numeric(row[col]) == 1)]
    return(paste(present_diseases, collapse = "|"))
  }
})

# 7. 整理與指定欄位順序
target_columns <- c(
  "subject_id","AP","PA","text","text_augment", "Finding.Labels",
  "No_Finding", "Atelectasis", "Cardiomegaly", "Consolidation", 
  "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", 
  "Infiltration", "Mass", "Nodule", "Pleural_Thickening", 
  "Pneumonia", "Pneumothorax", "Tuberculosis"
)

df_final <- df %>%
  select(all_of(intersect(target_columns, names(df))))

# 8. 匯出處理後的 CSV 檔案
write_csv(df_final, "C:/Users/Administrator/Desktop/mimic_cxr_aug_validate.csv")

mimic_cxr_aug_train <- read.csv("C:/Users/Administrator/Desktop/mimic_cxr_aug_train.csv")
mimic_cxr_aug_validate <- read.csv("C:/Users/Administrator/Desktop/mimic_cxr_aug_validate.csv")
df_final<-rbind(mimic_cxr_aug_train,mimic_cxr_aug_validate)
write_csv(df_final, "C:/Users/Administrator/Desktop/mimic_cxr_aug_all.csv")
