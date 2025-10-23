import pandas as pd

file1_path = "./0720/classified_format/sampled_th_0720_classified_format.csv"
file2_path = "./0720/classified_format/sampled_vn_0720_classified_format.csv"
target_path = "./0720/combined_classified_file.csv"

# 讀取兩個 CSV 文件
file1 = pd.read_csv(file1_path)
file2 = pd.read_csv(file2_path)

# 使用 concat 將兩個 DataFrame 垂直合併
combined_file = pd.concat([file1, file2], ignore_index=True)

# 顯示合併後的結果
print(combined_file)

# 將合併後的結果存成新的 CSV 文件
combined_file.to_csv(target_path, index=False)