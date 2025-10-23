import fasttext
import csv
import os
from tqdm import tqdm
from collections import defaultdict

def default_value():
    return 'spam'

if __name__ == '__main__':
    # 文件路徑
    date = "0713"
    language = "vn"

    input_csv_path = os.path.join(date, f"sampled_{language}_{date}.csv")
    output_folder = os.path.join(date,"classified_format")
    output_csv = f"sampled_{language}_{date}_classified_format.csv"
    output_csv_path = os.path.join(output_folder, output_csv)

    fast_text_model = fasttext.load_model('../models/FastText/lid.176.bin')

    source_languages = {
        "en": "en",
        "cn": "zh",
        "es": "es",
        "vn": "vi",
        "th": "th",
        "ko": "ko",
        "jp": "ja",
        "hi": "hi",
        "kh": "km",
        "br": "pt-br"
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 確定已處理行數
    processed_lines = 0
    if os.path.exists(output_csv_path):
        with open(output_csv_path, 'r', encoding='utf-8') as output_file:
            processed_lines = sum(1 for _ in output_file) - 1  # 減去標題行

    else:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(['id', 'text', 'category', 'label'])

    # 讀寫CSV文件
    with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # 讀取標題行

        # 計算總行數
        total_rows = sum(1 for _ in csv_reader)

        # 重置文件指針以重新讀取文件
        csv_file.seek(0)
        header = next(csv_reader)  # 讀取標題行

        # 跳過已處理的行
        for _ in range(processed_lines):
            next(csv_reader)

        with open(output_csv_path, 'a', newline='', encoding='utf-8') as output_file:
            csv_writer = csv.writer(output_file)

            current_id = processed_lines + 1  # 設定起始ID

            for row in tqdm(csv_reader, total=total_rows, initial=processed_lines, desc="Processing rows", unit="row"):
                message_content = row[6].replace('\n', ' ').replace('\r', ' ')
                
                prediction = fast_text_model.predict(message_content, k=1)
                source_lang = prediction[0][0].replace('__label__', '')
                confidence = prediction[1][0]
                
                if confidence < 0.6 or source_lang.lower() not in source_languages.values():
                    source_lang = source_languages[row[5]] if row[5] != 'null' else 'en'
                

                text = f"[{source_lang}] {message_content}"

                # if row[9] == '':
                #     print(current_id)

                category_dict = defaultdict(default_value, {
                    'normal': 'normal',
                    'misinformation': 'spam',
                    'advertise': 'spam',
                    'threats': 'toxic',
                    'harassment & sexually': 'sexual abuse',
                })
                
                category = category_dict[row[9].lower()]

                label = False if 'Normal' in row[9] else True


                csv_writer.writerow([current_id, text, category, label])
                current_id += 1