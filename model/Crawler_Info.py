# Crawler_Info.py

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
import time

# ===== 日誌設定 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, '..')
LOG_PATH = os.path.join(LOG_DIR, 'crawler.log')

# 確保日誌目錄存在
os.makedirs(LOG_DIR, exist_ok=True)

# 設置日誌配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 設置文件處理器
file_handler = logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 設置控制台處理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 初始日誌訊息
logging.info("===== 爬蟲腳本啟動 =====")

# 設定路徑
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'database.db')
IMAGES_DIR = os.path.join(BASE_DIR, '..', 'static', 'images')

# 確保圖片目錄存在
os.makedirs(IMAGES_DIR, exist_ok=True)
logging.info(f"圖片儲存目錄設置為: {IMAGES_DIR}")

# 資料庫初始化
def init_db():
    logging.info("初始化資料庫連線")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 創建 listings 表格
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            price REAL,
            size REAL,
            age REAL,
            floors INTEGER,
            bedroom INTEGER,
            city TEXT,
            location TEXT,
            district TEXT,
            house_type TEXT,
            pattern TEXT,
            tags TEXT,
            environment TEXT,
            url TEXT,
            image TEXT,
            CP_value INTEGER DEFAULT 0
        )
    ''')
    # 創建 coordinates 表格
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordinates (
            id INTEGER PRIMARY KEY,
            latitude REAL,
            longitude REAL
        )
    ''')
    conn.commit()
    logging.info("資料庫初始化完成")
    return conn

@dataclass
class HouseData:
    id: int
    title: str
    description: str
    price: float
    size: float
    age: float
    floors: int
    bedroom: int
    city: str
    location: str
    district: str
    house_type: str
    pattern: str
    tags: str
    environment: str
    url: str
    image: str
    CP_value: int = 0

    def to_tuple(self):
        return (
            self.id,
            self.title,
            self.description,
            self.price,
            self.size,
            self.age,
            self.floors,
            self.bedroom,
            self.city,
            self.location,
            self.district,
            self.house_type,
            self.pattern,
            self.tags,
            self.environment,
            self.url,
            self.image,
            self.CP_value
        )

# 初始化資料庫連線
conn = init_db()
cursor = conn.cursor()

# 總記錄數量
def get_total_listings():
    cursor.execute("SELECT COUNT(*) FROM listings")
    total = cursor.fetchone()[0]
    logging.info(f"目前 listings 表中有 {total} 筆記錄")
    return total

# 刪除 id > 500 的所有記錄
def delete_excess_records():
    logging.info("開始刪除 listings 表中 id > 500 的記錄")
    cursor.execute("DELETE FROM listings WHERE id > 500")
    conn.commit()
    logging.info("已刪除 listings 表中 id > 500 的所有記錄")
    
    # 同時將 coordinates 中對應的 id 的 latitude 和 longitude 設為 NULL
    logging.info("開始將 coordinates 表中 id > 500 的 latitude 和 longitude 設為 NULL")
    cursor.execute("UPDATE coordinates SET latitude = NULL, longitude = NULL WHERE id > 500")
    conn.commit()
    logging.info("已將 coordinates 表中 id > 500 的 latitude 和 longitude 設為 NULL")

# 獲取下一個可用的 ID (循環覆蓋)
def get_next_id(current_count):
    if current_count < 500:
        return current_count + 1
    else:
        # 從 1 開始覆蓋，循環至 500
        return ((current_count) % 500) + 1

# 更新現有資料並設置 CP_value 為 0
def update_record(house: HouseData):
    logging.info(f"更新 listings 表中 ID {house.id} 的資料")
    cursor.execute('''
        UPDATE listings
        SET title = ?, description = ?, price = ?, size = ?, age = ?, floors = ?, bedroom = ?,
            city = ?, location = ?, district = ?, house_type = ?, pattern = ?, tags = ?,
            environment = ?, url = ?, image = ?, CP_value = 0
        WHERE id = ?
    ''', (
        house.title, house.description, house.price, house.size, house.age, house.floors,
        house.bedroom, house.city, house.location, house.district, house.house_type,
        house.pattern, house.tags, house.environment, house.url, house.image,
        house.id
    ))
    conn.commit()
    logging.info(f"已更新 listings 表中 ID {house.id} 的資料，並將 CP_value 設為 0")

# 插入新資料
def insert_record(house: HouseData):
    logging.info(f"插入 listings 表中新資料，ID: {house.id}")
    cursor.execute('''
        INSERT INTO listings (
            id, title, description, price, size, age, floors, bedroom, city,
            location, district, house_type, pattern, tags, environment, url, image, CP_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', house.to_tuple())
    conn.commit()
    logging.info(f"已插入 listings 表中新資料，ID: {house.id}")

# 重置 coordinates 表中對應 id 的 latitude 和 longitude 為 NULL
def reset_coordinates(id):
    logging.info(f"重置 coordinates 表中 ID {id} 的 latitude 和 longitude 為 NULL")
    cursor.execute('''
        UPDATE coordinates
        SET latitude = NULL, longitude = NULL
        WHERE id = ?
    ''', (id,))
    conn.commit()
    logging.info(f"已將 coordinates 表中 ID {id} 的 latitude 和 longitude 設為 NULL")

# 下載圖片
def download_image(image_url, listing_title):
    """
    下載圖片並儲存到 static/images 資料夾。
    回傳儲存的圖片檔案名稱。
    """
    # 生成圖片檔案名稱，避免重複
    image_ext = os.path.splitext(image_url)[1].split('?')[0]  # 取得副檔名
    hash_object = hashlib.md5(image_url.encode())
    image_name = f"{hash_object.hexdigest()}{image_ext}"
    image_path = os.path.join(IMAGES_DIR, image_name)

    # 檢查圖片是否已存在，避免重複下載
    if os.path.exists(image_path):
        logging.info(f"圖片已存在，跳過下載: {image_name}")
        return image_name

    # 下載圖片，最多重試3次
    retries = 3
    for attempt in range(retries):
        try:
            logging.info(f"嘗試下載圖片: {image_url} (嘗試 {attempt + 1}/{retries})")
            img_resp = requests.get(image_url, headers=headers, timeout=10)
            if img_resp.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(img_resp.content)
                logging.info(f"已下載圖片: {image_name}")
                return image_name
            else:
                logging.warning(f"無法下載圖片: {image_url}，狀態碼: {img_resp.status_code}")
        except Exception as e:
            logging.error(f"下載圖片失敗: {image_url}, 錯誤: {e}")
        time.sleep(2)  # 等待2秒後重試

    logging.error(f"下載圖片最終失敗: {image_url}")
    return ''  # 若下載失敗，返回空字串

# 設定爬蟲相關參數
page_url = 'https://www.rakuya.com.tw/rent/rent_search?search=city&city=99&upd=1&page='
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://www.rakuya.com.tw'
}

@dataclass
class CrawlerResult:
    titles: list
    links: list

def crawler_url_set(url, page=1):
    link_list = []
    title_list = []

    for i in range(1, page + 1):
        current_page_url = f"{url}{i}"
        logging.info(f"爬取頁面: {current_page_url}")
        try:
            resp = requests.get(current_page_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logging.warning(f"無法訪問頁面: {current_page_url}，狀態碼: {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            items = soup.select('h6 > a[href*="/rent/rent_item"]')
            logging.info(f"找到 {len(items)} 個房源連結")
            for item in items:
                link = item.get('href')
                if not link:
                    continue
                title = item.get_text(strip=True)
                # 確保連結是完整的 URL
                full_link = f"https://www.rakuya.com.tw{link}" if not link.startswith('http') else link
                link_list.append(full_link)
                title_list.append(title)
        except Exception as e:
            logging.error(f"爬取頁面失敗: {current_page_url}, 錯誤: {e}")
        time.sleep(1)  # 為了避免過快爬取，加入延遲

    return CrawlerResult(titles=title_list, links=link_list)

def crawler_house_info(raw_title, link, overwrite_id):
    if not raw_title or not link:
        logging.warning("標題或連結為空，跳過此房源")
        return

    logging.info(f"開始處理房源: {raw_title} (ID: {overwrite_id})")
    try:
        resp = requests.get(link, headers=headers, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"無法訪問房源頁面: {link}，狀態碼: {resp.status_code}")
            return
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        logging.error(f"訪問房源頁面失敗: {link}, 錯誤: {e}")
        return

    # 嘗試抓資料
    content = soup.find('script', string=lambda s: s and 'window.tmpDataLayer' in s)
    if not content:
        logging.warning(f"找不到資料層腳本，跳過: {raw_title}")
        return
    data = content.string.strip()

    # 解析資料
    def extract_regex(pattern, data, default=''):
        match = re.search(pattern, data)
        return match.group(1) if match else default

    try:
        price = float(extract_regex(r'"price":(\d+)', data, '0'))
        age = float(extract_regex(r'"age":(\d+\.?\d*)', data, '0'))
        size = float(extract_regex(r'"item_variant":(\d+\.?\d*)', data, '0'))
        floor = int(extract_regex(r'"object_floor":(\d+)', data, '0'))
        bedroom = int(extract_regex(r'"bedrooms":(\d+)', data, '0'))
        city = extract_regex(r'"item_category":"(.*?)"', data, '')
        htype = extract_regex(r'"item_category5":"(.*?)"', data, '')
        tags = extract_regex(r'"object_tag":"(.*?)"', data, '').replace(',', ';')
        environment = extract_regex(r'"environment":"(.*?)"', data, '').replace(',', ';')
    except Exception as e:
        logging.error(f"解析資料失敗: {e}，房源: {raw_title}")
        return

    # 地址
    address_tag = soup.find('h1', class_='txt__address')
    location = address_tag.get_text(strip=True) if address_tag else ''

    # 格局, 環境
    pattern = ''
    environment_list = []
    li_elems = soup.find_all('li')
    for li in li_elems:
        label = li.find('span', class_='list__label')
        if label and "格局" in label.text:
            pattern = li.find('span', class_='list__content').get_text(strip=True)
        elif label and "物件環境" in label.text:
            b_tags = li.find('span', class_='list__content').find_all('b')
            environment_list = [b.get_text(strip=True) for b in b_tags]
    environment = ';'.join(environment_list)

    # 提取圖片 URL
    image_url = ''
    # 首先嘗試從 meta 標籤中提取
    meta_image = soup.find('meta', property='og:image')
    if meta_image and meta_image.get('content'):
        image_url = meta_image.get('content')
    else:
        # 如果找不到，嘗試通過 class="size-full object-contain" 的 img 標籤
        image_tag = soup.select_one('img.size-full.object-contain')
        if image_tag and image_tag.get('src'):
            image_url = image_tag.get('src')
        else:
            # 如果還找不到，嘗試其他方式提取圖片
            image_container = soup.find('div', class_='image-container')
            if image_container:
                image_tag = image_container.find('img')
                if image_tag and image_tag.get('src'):
                    image_url = image_tag.get('src')

    # 下載圖片並儲存
    image_name = ''
    if image_url:
        image_name = download_image(image_url, raw_title)
    else:
        logging.warning(f"找不到圖片 URL，房源: {raw_title}")

    # 描述
    description_tag = soup.find('div', class_='description')
    description = description_tag.get_text(strip=True) if description_tag else ''

    # 封裝
    house_obj = HouseData(
        id=overwrite_id,
        title=raw_title,
        description=description,
        price=price,
        size=size,
        age=age,
        floors=floor,
        bedroom=bedroom,
        city=city,
        location=location,
        district='',  # 如果有區域資料，請根據實際情況提取
        house_type=htype,
        pattern=pattern,
        tags=tags,
        environment=environment,
        url=link,
        image=image_name
    )

    # 決定是插入新資料還是更新現有資料
    cursor.execute("SELECT COUNT(*) FROM listings WHERE id = ?", (house_obj.id,))
    exists = cursor.fetchone()[0]
    if exists:
        logging.info(f"ID {house_obj.id} 已存在，準備更新")
        update_record(house_obj)
        reset_coordinates(house_obj.id)  # 覆蓋時將 coordinates 設為 NULL
    else:
        logging.info(f"ID {house_obj.id} 不存在，準備插入新資料")
        insert_record(house_obj)

def main():
    start_time = datetime.now()
    logging.info("爬蟲開始執行")

    # 讀取現有的記錄數量
    total_existing = get_total_listings()

    # 如果超過 500 筆，刪除 id > 500 的所有記錄
    if total_existing > 500:
        delete_excess_records()
        total_existing = get_total_listings()
        logging.info(f"刪除後，listings 表中有 {total_existing} 筆記錄")

    # 設定要爬取的頁數
    page_count = 10
    logging.info(f"開始爬取 {page_count} 頁的房源連結")
    contain = crawler_url_set(page_url, page_count)
    logging.info(f"共找到 {len(contain.titles)} 筆房源連結")

    for idx, (title, link) in enumerate(zip(contain.titles, contain.links), start=1):
        # 計算要覆蓋的 ID，從1開始循環
        if total_existing < 500:
            next_id = total_existing + 1
            total_existing += 1
        else:
            # 從 1 到 500 循環覆蓋
            next_id = ((idx - 1) % 500) + 1
        logging.info(f"處理第 {idx} 筆房源，ID: {next_id}")
        crawler_house_info(title, link, next_id)
        # 為了避免過快爬取，加入短暫的延遲
        time.sleep(1)  # 等待1秒

    end_time = datetime.now()
    logging.info(f"爬蟲完成，花費時間: {end_time - start_time}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.critical(f"腳本遇到未處理的錯誤: {e}")
    finally:
        conn.close()
        logging.info("資料庫連線已關閉")
        logging.info("===== 爬蟲腳本結束 =====")
