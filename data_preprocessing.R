library(dplyr)
library(tidyr)

# 讀取資料
df <- read.csv("D:/archive/Data_Entry_2017.csv", stringsAsFactors = FALSE)

# 定義目標欄位名稱與順序
target_labels <- c(
  "No Finding",         
  "Atelectasis", "Cardiomegaly", "Consolidation",     
  "Edema", "Effusion", "Emphysema",              
  "Fibrosis", "Hernia", "Infiltration",              
  "Mass", "Nodule", "Pleural_Thickening",      
  "Pneumonia", "Pneumothorax"    
)

# 處理流程：
df_processed <- df %>%
  mutate(
    row_id = row_number(),
    split_label = Finding.Labels                           # 1. 複製一份欄位專門拿來拆分，保留原 Finding.Labels
  ) %>%  
  separate_rows(split_label, sep = "\\|") %>%             # 2. 將複製的欄位拆成多列
  mutate(value = 1) %>%                                    # 3. 建立標記欄位
  pivot_wider(
    names_from = split_label, 
    values_from = value, 
    values_fill = 0
  ) %>% 
  select(-row_id) %>%                                     # 4. 移除暫存 ID
  relocate(all_of(target_labels), .after = last_col())    # 5. 將疾病欄位依照指定順序移到資料集最後方

colnames(df_processed)[colnames(df_processed)=="No Finding"]<-"No_Finding"

# 寫入新檔案
write.csv(df_processed, "D:/archive/Data_Entry_2017_processed.csv", row.names = FALSE)

df <- read.csv("D:/archive/Data_Entry_2017_processed.csv", stringsAsFactors = FALSE)
