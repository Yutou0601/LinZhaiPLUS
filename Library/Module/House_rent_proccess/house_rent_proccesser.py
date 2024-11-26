# house_rent_processor.py

import pandas as pd
import numpy as np
import os

# 定義台北市各區的名稱及其對應的編號
district_keyword_mapping = {
    '松山': '松山區',
    '信義': '信義區',
    '大安': '大安區',
    '中山': '中山區',
    '中正': '中正區',
    '大同': '大同區',
    '萬華': '萬華區',
    '文山': '文山區',
    '南港': '南港區',
    '內湖': '內湖區',
    '士林': '士林區',
    '北投': '北投區'
}

district_mapping = {
    '松山區': 1,
    '信義區': 2,
    '大安區': 3,
    '中山區': 4,
    '中正區': 5,
    '大同區': 6,
    '萬華區': 7,
    '文山區': 8,
    '南港區': 9,
    '內湖區': 10,
    '士林區': 11,
    '北投區': 12,
    '未知區域': 0  # 為未知區域分配編號 0
}

def extract_district(name, keyword_mapping):
    """
    從房屋名稱中提取區域名稱。
    如果沒有找到對應的區域，返回 '未知區域'。
    """
    for keyword, full_district in keyword_mapping.items():
        if keyword in name:
            return full_district
    return '未知區域'

def process_house_rent_data(input_csv, output_csv):
    """
    處理房屋租賃資料，提取區域並映射到編號，保存到新的 CSV 文件。
    """
    # 讀取資料
    try:
        data = pd.read_csv(input_csv, encoding='utf-8')
    except FileNotFoundError:
        print(f"錯誤: 找不到文件 {input_csv}")
        return
    except Exception as e:
        print(f"讀取文件時出錯: {e}")
        return

    # 顯示原始資料的前幾行
    print("原始 House_Rent_Info.csv 的前幾行：")
    print(data.head())

    # 檢查必要的欄位是否存在
    required_columns = ['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags']
    for col in required_columns:
        if col not in data.columns:
            print(f"錯誤: 缺少必要的欄位 '{col}'")
            return

    # 提取區域名稱
    data['District'] = data['Name'].apply(lambda x: extract_district(x, district_keyword_mapping))

    # 映射區域到編號
    data['District_Index'] = data['District'].map(district_mapping)

    # 檢查是否有未映射的區域
    unmapped = data[data['District_Index'].isnull()]
    if not unmapped.empty:
        print("警告: 以下區域未被映射到編號，將被標記為 '未知區域' 並賦予編號 0")
        print(unmapped[['ID', 'Name', 'District']])
        data.loc[data['District_Index'].isnull(), 'District'] = '未知區域'
        data.loc[data['District_Index'].isnull(), 'District_Index'] = 0

    # 確保所有 District_Index 都是整數
    data['District_Index'] = data['District_Index'].astype(int)

    # 顯示處理後的資料前幾行
    print("\n處理後的資料集前幾行：")
    print(data.head())

    # 選擇需要的欄位順序
    output_columns = ['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags', 'District', 'District_Index']

    # 保存到新的 CSV 文件
    try:
        data.to_csv(output_csv, columns=output_columns, index=False, encoding='utf-8-sig')
        print(f"\n成功保存處理後的資料到 {output_csv}")
    except Exception as e:
        print(f"保存文件時出錯: {e}")

if __name__ == "__main__":
    # 定義輸入和輸出文件的路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(current_dir, 'House_Rent_Info.csv')
    output_csv = os.path.join(current_dir, 'New_House_Rent_Info.csv')

    # 處理資料
    process_house_rent_data(input_csv, output_csv)
