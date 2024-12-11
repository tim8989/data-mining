# import pandas as pd
# import ast
import argparse

# config處理
parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--pre_dataset_name',metavar='-d',type=str,required=False,default='0000.csv',help="preprocessing dataset name")
parser.add_argument('--pre_dataset_folder', metavar='-o', type=str, required=False, default='./your_dataset')

args = parser.parse_args()

pre_dataset_name = args.pre_dataset_name
pre_dataset_folder = args.pre_dataset_folder

# 讀取原始 CSV 文件
input_file = 'dataset/' + pre_dataset_folder + '/' + pre_dataset_name
formate_file = 'dataset/' + pre_dataset_folder + '/output.csv'
output_label_file = 'dataset/' + pre_dataset_folder + '/interpretation_label.csv'

# dataset格式轉為 -> star_id, time_array, brightness_array
def formate_dataset(input_file):
    data = pd.read_csv(input_file)

    # 初始化新的 DataFrame
    output_data = {
        "star_id": [],
        "time": [],
        "brightness": []
    }
    
    # 遍歷每一列，提取時間和亮度
    for index, row in data.iterrows():
        star_id = row['object_id']
        
        # 將儲存為字串的二維陣列轉為 Python 列表
        time_array = ast.literal_eval(row['times_wv'])
        brightness_array = ast.literal_eval(row['lightcurve'])
        
        # 取出每個元素的第 0 項
        times = [element[0] for element in time_array]

        # timeslot 0 的不取
        valid_indices = [i for i, t in enumerate(times) if t != 0]

        brightnesses = [element[1] for element in brightness_array]
        
        # 將資料加入到新的 DataFrame
        for i in valid_indices:
            output_data['star_id'].append(star_id)
            output_data['time'].append(times[i])
            output_data['brightness'].append(brightnesses[i])

    # 將結果轉為 DataFrame 並存成新的 CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(formate_file, index=False)

    print(f"生成的文件已儲存為 {formate_file}")

def label(formate_file):
    #讀取 CSV 檔案
    df = pd.read_csv(formate_file)

    # 1. 新增 'label' 欄位並初始化為 0
    df['label'] = 0

    # 2. 計算每個 star_id 的平均亮度和標準差
    stats = df.groupby('star_id')['brightness'].agg(
        mean='mean',  # 計算平均亮度
        std='std'     # 計算標準差
    )

    # 3. 根據條件設置 label 欄位的值
    def set_label(row):
        # 從 stats 中獲取該 star_id 的平均亮度和標準差
        mean = stats.loc[row['star_id'], 'mean']
        std = stats.loc[row['star_id'], 'std']
        
        # 根據亮度小於平均值減去三倍標準差或大於平均值加三倍標準差來設置 label
        if row['brightness'] < (mean - 3 * std) or row['brightness'] > (mean + 3 * std):
            return 1
        return 0

    df['label'] = df.apply(set_label, axis=1)

    # 儲存處理後的結果到新 CSV 檔案（選擇性）
    df.to_csv(output_label_file, index=False)

    # 顯示結果
    print(f"生成的文件已儲存為 {output_label_file}")

formate_dataset(input_file)
label(formate_file)

# 未來需處理
# - 需要取出 幾顆固定的星星 出現在 N 個連續的 timeslot 的資料
# - 再將取出的資料分成 test / train 轉成 .txt (可以參考如下)