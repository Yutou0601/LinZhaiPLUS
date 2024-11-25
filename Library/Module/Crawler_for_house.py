import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# 定義房屋資料類別
class HouseData:
    def __init__(self, id, name, price, size, age, floor, location, tags):
        self.id = id
        self.name = name
        self.price = price
        self.size = size
        self.age = age
        self.floor = floor
        self.location = location
        self.tags = tags

    def to_dict(self):
        return {
            'ID': self.id,
            'Name': self.name,
            'Price': self.price,
            'Size': self.size,
            'Age': self.age,
            'Floor': self.floor,
            'Location': self.location,
            'Tags': ','.join(self.tags) if self.tags else ""
        }

# 從房屋列表頁面提取房屋詳細資訊連結
def fetch_house_links(base_url, max_pages=1):
    house_links = []
    for i in range(1, max_pages + 1):
        url = base_url.replace('i', str(i))
        response = requests.get(url)
        if response.status_code != 200:
            print(f"第 {i} 頁加載失敗，狀態碼: {response.status_code}")
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'/sell_item/info\?ehid=\w+'))
        for link in links:
            full_url = 'https://www.rakuya.com.tw' + link['href'] if not link['href'].startswith('http') else link['href']
            house_links.append(full_url)
        time.sleep(random.uniform(1, 3))  # 隨機延遲
    return list(set(house_links))  # 去重

# 過濾已存在的房屋連結
def filter_existing_links(house_links, filename='House_Rent_Info.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path)
            existing_ids = set(df['ID'])
        except pd.errors.EmptyDataError:
            print(f"{filename} 是空檔案，將忽略已存在的檢查。")
            existing_ids = set()
    else:
        existing_ids = set()

    filtered_links = [link for link in house_links if re.search(r'ehid=(\w+)', link).group(1) not in existing_ids]
    return filtered_links


# 從房屋詳細資訊頁面提取所需資訊
def fetch_house_data(house_url):
    try:
        response = requests.get(house_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"無法訪問 {house_url}，錯誤: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        id_match = re.search(r'ehid=(\w+)', house_url)
        house_id = id_match.group(1) if id_match else None

        content = soup.find('script', string=lambda string: string and 'window.tmpDataLayer' in string)
        if content:
            data = content.string.strip()
            name = re.search(r'"item_name":"(.*?)"', data)
            price = re.search(r'"price":(\d+)', data)
            age = re.search(r'"age":(\d+\.?\d*)', data)
            size = re.search(r'"object_main_size":(\d+\.?\d*)', data)
            floor = re.search(r'"object_floor":(\d+)', data)
            location = re.search(r'"item_category":"(.*?)"', data)
            tags = re.search(r'"object_tag":"(.*?)"', data)

            name = name.group(1) if name else "未知名稱"
            price = int(price.group(1)) if price else None
            age = float(age.group(1)) if age else None
            size = float(size.group(1)) if size else None
            floor = int(floor.group(1)) if floor else None
            location = location.group(1) if location else "未知位址"
            tags = tags.group(1).split(',') if tags else []

            return {
                'ID': house_id,
                'Name': name,
                'Price': price,
                'Size': size,
                'Age': age,
                'Floor': floor,
                'Location': location,
                'Tags': ','.join(tags) if tags else ""
            }
        else:
            print(f"無法從 {house_url} 提取 JavaScript 資料")
            return None
    except Exception as e:
        print(f"解析 {house_url} 時出錯: {e}")
        return None

# 並行抓取所有房屋詳細資訊
def fetch_all_house_data(house_links, max_workers=10):
    house_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_house_data, url): url for url in house_links}
        for future in as_completed(future_to_url):
            try:
                data = future.result()
                if data:
                    house_data.append(data)
            except Exception as e:
                print(f"抓取資料時出錯: {e}")
    return house_data

# 更新 CSV 的函數
def update_csv(house_data, filename='House_Rent_Info.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"{filename} 是空檔案，初始化為空 DataFrame")
            df = pd.DataFrame(columns=['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags'])
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags'])

    # 移除資料缺失的記錄
    house_data = [
        house for house in house_data
        if house['Name'] != "未知名稱" and house['Price'] is not None and house['Size'] is not None
    ]

    # 檢查現有資料
    existing_data = df.set_index('ID')['Price'].to_dict() if not df.empty else {}
    new_data = []
    for house in house_data:
        if house['ID'] in existing_data:
            if house['Price'] == existing_data[house['ID']]:
                continue  # 價格相同，跳過
        new_data.append(house)

    # 合併新資料
    if new_data:
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

    # 保存資料到檔案
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"數據已保存至 {file_path}")


# 主程式
if __name__ == "__main__":
    BASE_URL = "https://www.rakuya.com.tw/sell/result?city=15&sort=11&page=i"
    TOTAL_PAGES = 100  # 爬取頁數
    house_links = fetch_house_links(BASE_URL, max_pages=TOTAL_PAGES)
    print(f"共找到 {len(house_links)} 個房屋連結。")

    house_links = filter_existing_links(house_links, filename='House_Rent_Info.csv')
    print(f"過濾後剩餘 {len(house_links)} 個需抓取的連結。")

    house_data = fetch_all_house_data(house_links, max_workers=20)
    print(f"成功提取 {len(house_data)} 筆房屋詳細資料。")

    update_csv(house_data, filename='House_Rent_Info.csv')
