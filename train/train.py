import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import datetime


learning_rate = 2e-6
epochs = 10
batch_size_train = 16
batch_size_eval = 32

test_file = ".."
test_file = ".."

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 初始化wandb，使用正確的project
wandb.init(project="ChatFilter - Multilingual Bert", name=current_time)

# 設定wandb的訓練參數
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size_train": batch_size_train,
    "batch_size_eval": batch_size_eval
}

print("開始數據處理和模型訓練流程...")

# 設定CUDA設備
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型訓練將使用設備: {device}")

# 文件路徑
dataset_dir = '../dataset/cache/'

# 確保目錄存在
os.makedirs(dataset_dir, exist_ok=True)

tokenizer_path = "../models/tokenizer/vocab_expanded_tokenizer"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# tokenizer = BertTokenizer.from_pretrained('unitary/toxic-bert', trust_remote_code=True, cache_dir=tokenizer_path)
# tokenizer = AutoTokenizer.from_pretrained('aisingapore/sealion-bert-base', trust_remote_code=True, cache_dir=cache_dir)

# 加載數據集
if os.path.exists(train_tokenized_file) and os.path.exists(val_tokenized_file) and os.path.exists(test_th_tokenized_file) and os.path.exists(test_vn_tokenized_file):
    print("階段1: 加載預編碼數據...")
    train_tokenized = torch.load(train_tokenized_file)
    val_tokenized = torch.load(val_tokenized_file)
    test_th_tokenized = torch.load(test_th_tokenized_file)
    test_vn_tokenized = torch.load(test_vn_tokenized_file)
    
    train_labels = torch.load(train_labels_file)
    val_labels = torch.load(val_labels_file)
    test_th_labels = torch.load(test_th_labels_file)
    test_vn_labels = torch.load(test_vn_labels_file)
    print("數據加載完成。")
else:
    print("階段1: 加載原始數據並進行分割...")
    # file_path = '../dataset/0720/combined/combined_classified_file.csv'
    folder_path = '/mnt/user_ludwig/ChatFilter/csv/'
    file_name = '2024-07-'
    file_paths = [os.path.join(folder_path, file_name + str(i) + '.csv') for i in range(11, 16)]
    
    dataset = pd.DataFrame()
    for csv_files in file_paths:
        df = pd.read_csv(csv_files).loc[:, ['language', 'messageContent', 'label']]
        dataset = pd.concat([dataset, df])

    # df = pd.read_csv(file_path)
    print(f"數據加載完成。數據集大小: {dataset.shape}")
    
    # 分割數據集
    print("階段2: 分割數據集...")
    train_df, val_df = train_test_split(dataset, test_size=0.1, random_state=42, stratify=dataset['label'])
    test_th_df = pd.read_csv(test_th_file)
    test_vn_df = pd.read_csv(test_vn_file)

    print(f"數據集分割完成。訓練集大小: {train_df.shape}, 驗證集大小: {val_df.shape}")
    
    # 檢視各數據集label分佈狀況
    print("訓練集標籤分佈:")
    print(train_df['label'].value_counts())
    
    print("驗證集標籤分佈:")
    print(val_df['label'].value_counts())

    # 轉換標籤
    print("階段3: 轉換標籤...")
    for df in [train_df, val_df, test_th_df, test_vn_df]:
        df['label'] = df['label'].apply(lambda x: 1 if x == True else 0)
    print("標籤轉換完成。")
    
    # 轉換後再次檢查標籤分佈
    print("轉換標籤後訓練集標籤分佈:")
    print(train_df['label'].value_counts())
    
    print("轉換標籤後驗證集標籤分佈:")
    print(val_df['label'].value_counts())
    
    # 加載標記器
    print("階段4: 加載標記器...")
    
    print("標記器加載完成。")
    
    # Tokenization
    print("階段5: 文本編碼...")
    def tokenize_function(texts):
        return tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    tokenized = {}

    for name, df in [("train", train_df), ("val", val_df), ("test_th", test_th_df), ("test_vn", test_vn_df)]:
        print(f"正在編碼{name}...")
        texts = df['messageContent'].tolist()
        
        # 使用批處理來提高效率
        batch_size = 64
        encoded = {'input_ids': [], 'attention_mask': []}
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"{name}編碼進度", total=(len(texts) + batch_size - 1) // batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_tokenized = tokenize_function(batch_texts)
            encoded['input_ids'].append(batch_tokenized['input_ids'])
            encoded['attention_mask'].append(batch_tokenized['attention_mask'])
        
        encoded = {k: torch.cat(v) for k, v in encoded.items()}
        tokenized[f"{name}_tokenized"] = encoded
    
    print("文本編碼完成。")

    # 保存編碼數據和標籤
    print("階段6: 保存編碼數據和標籤...")
    torch.save(tokenized['train_tokenized'], train_tokenized_file)
    torch.save(tokenized['val_tokenized'], val_tokenized_file)
    torch.save(tokenized['test_th_tokenized'], test_th_tokenized_file)
    torch.save(tokenized['test_vn_tokenized'], test_vn_tokenized_file)
    
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    test_th_labels = test_th_df['label'].tolist()
    test_vn_labels = test_vn_df['label'].tolist()
    
    torch.save(train_labels, train_labels_file)
    torch.save(val_labels, val_labels_file)
    torch.save(test_th_labels, test_th_labels_file)
    torch.save(test_vn_labels, test_vn_labels_file)
    print("保存完成。")

    # 加載保存後的數據
    train_tokenized = torch.load(train_tokenized_file)
    val_tokenized = torch.load(val_tokenized_file)
    test_th_tokenized = torch.load(test_th_tokenized_file)
    test_vn_tokenized = torch.load(test_vn_tokenized_file)
    
    train_labels = torch.load(train_labels_file)
    val_labels = torch.load(val_labels_file)
    test_th_labels = torch.load(test_th_labels_file, weights_only=True)
    test_vn_labels = torch.load(test_vn_labels_file, weights_only=True)

# 確保變量定義
assert 'train_tokenized' in locals(), "train_tokenized 未定義"
assert 'val_tokenized' in locals(), "val_tokenized 未定義"
assert 'test_th_tokenized' in locals(), "test_th_tokenized 未定義"
assert 'test_vn_tokenized' in locals(), "test_vn_tokenized 未定義"
assert 'train_labels' in locals(), "train_labels 未定義"
assert 'val_labels' in locals(), "val_labels 未定義"
assert 'test_th_labels' in locals(), "test_th_labels 未定義"
assert 'test_vn_labels' in locals(), "test_vn_labels 未定義"

# 數據集類定義和創建
print("階段7: 創建數據集...")
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized, labels):
        self.tokenized = tokenized
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_tokenized, train_labels)
val_dataset = ToxicDataset(val_tokenized, val_labels)
test_th_dataset = ToxicDataset(test_th_tokenized, test_th_labels)
test_vn_dataset = ToxicDataset(test_vn_tokenized, test_vn_labels)
print("數據集創建完成。")

# 加載模型
model_path = "../models/huggingface/model/"
print("階段8: 加載模型...")
model = BertForSequenceClassification.from_pretrained('unitary/toxic-bert', num_labels=1, ignore_mismatched_sizes=True, cache_dir=model_path)
model.resize_token_embeddings(len(tokenizer))

model.to(device)  # 將模型移到 GPU（如果可用）
print("模型加載完成。")

# 訓練參數設置
print("階段9: 設置訓練參數...")
output_dir = f'../models/model/toxic_bert/fine_tuned/{current_time}'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size=batch_size_eval,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["wandb"],
    gradient_accumulation_steps=2,
    fp16=True if torch.cuda.is_available() else False,
    save_total_limit=2  # 保留最好的和最后一个模型
)


print("訓練參數設置完成。")

# 計算總訓練步數
num_training_steps = len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print(f"總訓練步數: {num_training_steps}")

# 創建優化器和學習率調度器
print("階段10: 創建優化器和學習率調度器…")
def create_optimizer_and_scheduler(model, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

# 定義評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 創建 EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=100)
print("早停回調創建完成。")

# 定義Trainer
print("階段11: 創建Trainer…")

trainer = Trainer(
    model=model,                         # the initialized model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    tokenizer=tokenizer,                 # tokenizer (sealion-bert)
    optimizers=create_optimizer_and_scheduler(model, num_training_steps),
    compute_metrics=compute_metrics,    # evaluation metrics
    callbacks=[early_stopping_callback]
)


print("Trainer 創建完成。")

# 在訓練開始前進行初始評估
print("訓練開始前進行初始評估...")
initial_eval_results = trainer.evaluate(eval_dataset=val_dataset)
print("初始評估結果:", initial_eval_results)

# 開始訓練
print("階段12: 開始訓練…")
trainer.train()
print("訓練完成。")

# # 在測試集上評估模型
# print("階段13: 在測試集上評估模型…")
# test_th_results = trainer.evaluate(test_th_dataset)
# test_vn_results = trainer.evaluate(test_vn_dataset)

# print("泰語測試集結果:", test_th_results)
# print("越南語測試集結果:", test_vn_results)

print("整個流程完成。")