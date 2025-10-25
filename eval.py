from transformers import Trainer, BertTokenizer, BertForSequenceClassification, BertConfig
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import torch
import pandas as pd
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型訓練將使用設備: {device}")


# Initialize the tokenizer
tokenizer_path = "../models/tokenizer/vocab_expanded_tokenizer"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

model_path = '../models/model/toxic_bert/fine_tuned/20240919-052027/checkpoint-16860'
config_path = os.path.join(model_path, 'config.json')
model_tensor = os.path.join(model_path, 'pytorch_model.bin')

# 加載模型
if not os.path.exists(config_path) or not os.path.exists(model_tensor):
    raise FileNotFoundError(f"模型配置文件或 bin 文件不存在。請檢查路徑和文件名是否正確。")

config = BertConfig.from_json_file(config_path)

model = BertForSequenceClassification(config)

# Load the model's state_dict from the .bin file
state_dict = torch.load(model_tensor, map_location=device)

model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(state_dict)
model.to(device)  # 將模型移到 GPU（如果可用）
print("模型和標記器加載完成。")

def tokenize_function(texts):
        return tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# cache_path = '../dataset/cache/'
# tokenized_testing_data = os.path.join(cache_path)

# file_path = "/mnt/user_ludwig/ChatFilter/csv/2024-07-20.csv"
# dataset = pd.read_csv(file_path)


# dataset['label'] = dataset['label'].apply(lambda x: 1 if x == True else 0)
# texts = dataset['messageContent'].tolist()
# batch_size = 64
# encoded = {'input_ids': [], 'attention_mask': []}
# for i in tqdm(range(0, len(texts), batch_size), desc=f"編碼進度", total=(len(texts) + batch_size - 1) // batch_size):
#     batch_texts = texts[i:i+batch_size]
#     batch_tokenized = tokenize_function(batch_texts)
#     encoded['input_ids'].append(batch_tokenized['input_ids'])
#     encoded['attention_mask'].append(batch_tokenized['attention_mask'])

# encoded = {k: torch.cat(v) for k, v in encoded.items()}

# testing_labels = dataset['label'].tolist()

print("階段2：加載已編碼的測試數據開始...")
encoded_data_path = '../dataset/cache/'

test_th_encodings_path = os.path.join(encoded_data_path, 'test_th_tokenized.pt')
test_th_labels_path = os.path.join(encoded_data_path, 'test_th_labels.pt')
test_vn_encodings_path = os.path.join(encoded_data_path, 'test_vn_tokenized.pt')
test_vn_labels_path = os.path.join(encoded_data_path, 'test_vn_labels.pt')

if not os.path.exists(test_th_encodings_path) or not os.path.exists(test_th_labels_path):
    raise FileNotFoundError(f"th編碼數據文件不存在。請檢查路徑和文件名是否正確。")

if not os.path.exists(test_vn_encodings_path) or not os.path.exists(test_vn_labels_path):
    raise FileNotFoundError(f"vn編碼數據文件不存在。請檢查路徑和文件名是否正確。")

test_th_encodings = torch.load(test_th_encodings_path)
test_th_labels = torch.load(test_th_labels_path)  # 直接使用，无需 .tolist()
test_vn_encodings = torch.load(test_vn_encodings_path)
test_vn_labels = torch.load(test_vn_labels_path)  # 直接使用，无需 .tolist()

print("階段2：加載已編碼的測試數據結束。")

def evaluate_model(test_encodeds, test_labels, dataset_name, fp, tp, fn, tn):
    # 階段3：將測試數據移動到設備
    print(f"階段3：將{dataset_name}測試數據移動到設備開始...")
    input_ids = test_encodeds['input_ids'].to(device)
    attention_mask = test_encodeds['attention_mask'].to(device)
    print(f"階段3：將{dataset_name}測試數據移動到設備結束。")

    # 階段4：模型推論
    print(f"階段4：模型推論開始 ({dataset_name})...")
    model.eval()
    batch_size = 16  # 設置適當的批次大小
    all_predictions = []
    all_probabilities = []
    texts = []

    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), batch_size), desc=f"{dataset_name}模型推論進度"):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)
            
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.extend(probabilities)
            predictions = np.where(probabilities > 0.3, 1, 0).flatten()
            all_predictions.extend(predictions)

            # 解码文本
            batch_texts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            texts.extend(batch_texts)


    print(f"階段4：模型推論結束 ({dataset_name})。")

    # 階段5：計算混淆矩陣
    print(f"階段5：計算{dataset_name}混淆矩陣開始...")
    conf_matrix = confusion_matrix(test_labels, all_predictions)
    print(f"階段5：計算{dataset_name}混淆矩陣結束。")

    # 階段6：輸出分類報告
    print(f"階段6：輸出{dataset_name}分類報告開始...")
    class_report = classification_report(test_labels, all_predictions, output_dict=True)

    # 打印混淆矩陣
    print(f"{dataset_name}混淆矩陣:")
    conf_matrix_df = pd.DataFrame(conf_matrix)
    print(conf_matrix_df)

    # 打印分類報告
    print(f"\n{dataset_name}分類報告:")
    print(pd.DataFrame(class_report).transpose())

    print(f"階段6：輸出{dataset_name}分類報告結束。")
    
    # 階段7：生成並保存 ROC AUC 圖
    print(f"階段7：生成並保存 {dataset_name} ROC AUC 圖開始...")
    
    test_result_dir = 'results/'
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)    
        
    test_labels_array = np.array(test_labels)
    all_probabilities_array = np.array(all_probabilities).flatten()

    fpr, tpr, thresholds = roc_curve(test_labels_array, all_probabilities_array)
    roc_auc = roc_auc_score(test_labels_array, all_probabilities_array)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 在曲線中標注出一些重要的閾值及對應的 FPR 和 TPR
    for i in range(0, len(thresholds), max(1, len(thresholds)//10)): 
        label = f'Thresh: {thresholds[i]:.2f}\nTPR: {tpr[i]:.2f}, FPR: {fpr[i]:.2f}'
        plt.scatter(fpr[i], tpr[i], color='red') 
        plt.annotate(label, (fpr[i], tpr[i]), textcoords="offset points", xytext=(10, -10), ha='center', fontsize=8)    
        
        # 添加從紅點到軸的網格線
        plt.axhline(y=tpr[i], color='gray', linestyle='--', lw=0.5)
        plt.axvline(x=fpr[i], color='gray', linestyle='--', lw=0.5)
    
    # 計算並標注由 fp, tp, fn, tn 帶來的 FPR 和 TPR
    custom_fpr = fp / (fp + tn)
    custom_tpr = tp / (tp + fn)
    
    plt.scatter(custom_fpr, custom_tpr, color='blue', marker='x', s=100, label=f'Custom TPR: {custom_tpr:.2f}, FPR: {custom_fpr:.2f}')
    plt.annotate(f'LCS\nTPR: {custom_tpr:.2f}\nFPR: {custom_fpr:.2f}', (custom_fpr, custom_tpr), textcoords="offset points", xytext=(-50, -20), ha='center', fontsize=10, color='blue')    

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'./{test_result_dir}/roc_auc_curve_{dataset_name}.png')
    plt.close()
    print(f"階段7：生成並保存 {dataset_name} ROC AUC 圖結束。")

    # -------------------------------------------------------------
    # 保存推論結果至 CSV
    # -------------------------------------------------------------
    print(f"保存 {dataset_name} 推論結果至 CSV 檔案...")
    df = pd.DataFrame({
        'text': texts,
        'label': test_labels,
        'infer_score': all_probabilities_array
    })
    df.to_csv(f'./{test_result_dir}/{dataset_name}_inference_results.csv', index=False)
    print(f"{dataset_name} 推論結果已保存至 CSV 檔案。")
    
    # -------------------------------------------------------------
    # 保存 FPR, TPR 和 Thresholds 至新的 CSV 檔案
    # -------------------------------------------------------------
    print(f"保存 {dataset_name} 的 FPR, TPR 和 Thresholds 至新的 CSV 檔案...")
    roc_data_df = pd.DataFrame({
        'fpr': np.round(fpr, 4),
        'tpr': np.round(tpr, 4),
        'thresholds': np.round(thresholds, 4)
    })
    roc_data_csv_path = os.path.join(test_result_dir, f'{dataset_name}_roc_data.csv')
    roc_data_df.to_csv(roc_data_csv_path, index=False)
    print(f"{dataset_name} 的 FPR, TPR 和 Thresholds 已保存至新的 CSV 檔案。")

# 評估泰語測試集，假設你的FP, TP, FN, TN值如下
evaluate_model(test_th_encodings, test_th_labels, "TH", fp=666, tp=334, fn=16, tn=984)

# 評估越南語測試集，假設你的FP, TP, FN, TN值如下
evaluate_model(test_vn_encodings, test_vn_labels, "VN", fp=823, tp=177, fn=186, tn=814)

print("整個流程完成。")
print("整個流程完成。")