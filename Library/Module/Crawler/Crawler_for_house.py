import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from urllib.parse import urlparse, urljoin
import urllib.robotparser as robotparser
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 設定日誌記錄
logging.basicConfig(
    filename='scraper.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 生成隨機的 HTTP 標頭
USER_AGENTS = [
    # Chrome on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/93.0.4577.63 Safari/537.36',
    # Firefox on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0',
    # Safari on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)'
    ' Version/14.1.2 Safari/605.1.15',
    # Chrome on macOS
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/92.0.4515.159 Safari/537.36',
    # Edge on Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/92.0.4515.159 Safari/537.36 Edg/92.0.902.84',
    # iPhone Safari
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15'
    ' (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
    # Android Chrome
    'Mozilla/5.0 (Linux; Android 11; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko)'
    ' Chrome/92.0.4515.159 Mobile Safari/537.36',
]

COMMON_HEADERS = {
    'Accept-Language': 'zh-TW,zh;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.google.com/',
}

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

# 初始化 requests.Session() 並設定重試機制
def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = create_session()

# 生成隨機的 HTTP 標頭
def generate_headers():
    headers = COMMON_HEADERS.copy()
    headers['User-Agent'] = random.choice(USER_AGENTS)
    return headers

# 解析 robots.txt
def get_robot_parser(base_url, robots_txt_content):
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base_url, '/robots.txt'))
    rp.parse(robots_txt_content.splitlines())
    return rp

# 動態抓取 robots.txt
def fetch_robots_txt(base_url):
    robots_url = urljoin(base_url, '/robots.txt')
    try:
        headers = generate_headers()
        response = session.get(robots_url, headers=headers, timeout=10)
        response.raise_for_status()
        logging.info(f"成功抓取 robots.txt: {robots_url}")
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"無法抓取 robots.txt，錯誤: {e}")
        return ""

# 在請求之間根據 Crawl-delay 增加延遲
def sleep_delay(rp):
    crawl_delay = rp.crawl_delay("*")
    if crawl_delay is None:
        crawl_delay = 1  # 預設延遲為 1 秒
    time.sleep(crawl_delay + random.uniform(0, 0.5))  # 加入隨機性避免規律性

# 從房屋列表頁面提取房屋詳細資訊連結
def fetch_house_links(base_url, rp, max_pages=1):
    house_links = []
    for i in range(1, max_pages + 1):
        url = base_url.format(i)
        headers = generate_headers()
        if not rp.can_fetch("*", url):
            logging.warning(f"根據 robots.txt，無法抓取 URL: {url}")
            continue
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logging.info(f"成功抓取房屋列表頁面: {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"第 {i} 頁加載失敗，錯誤: {e}")
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'/sell_item/info\?ehid=\w+'))
        for link in links:
            href = link.get('href')
            full_url = urljoin(base_url, href) if not href.startswith('http') else href
            if rp.can_fetch("*", full_url):
                house_links.append(full_url)
            else:
                logging.warning(f"根據 robots.txt，無法抓取 URL: {full_url}")
        sleep_delay(rp)
    unique_links = list(set(house_links))  # 去重
    logging.info(f"共找到 {len(unique_links)} 個房屋連結。")
    return unique_links

# 過濾已存在的房屋連結
def filter_existing_links(house_links, filename='House_Rent_Info.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path)
            existing_ids = set(df['ID'])
            logging.info(f"已存在 {len(existing_ids)} 個房屋 ID。")
        except pd.errors.EmptyDataError:
            logging.warning(f"{filename} 是空檔案，將忽略已存在的檢查。")
            existing_ids = set()
    else:
        existing_ids = set()
        logging.info(f"{filename} 不存在，將抓取所有連結。")

    filtered_links = []
    for link in house_links:
        match = re.search(r'ehid=(\w+)', link)
        if match:
            house_id = match.group(1)
            if house_id not in existing_ids:
                filtered_links.append(link)
    logging.info(f"過濾後剩餘 {len(filtered_links)} 個需抓取的連結。")
    return filtered_links

# 從房屋詳細資訊頁面提取所需資訊
def fetch_house_data(house_url, rp):
    headers = generate_headers()
    if not rp.can_fetch("*", house_url):
        logging.warning(f"根據 robots.txt，無法抓取 URL: {house_url}")
        return None
    try:
        response = session.get(house_url, headers=headers, timeout=10)
        response.raise_for_status()
        logging.info(f"成功抓取房屋詳細頁面: {house_url}")
    except requests.exceptions.Timeout:
        logging.error(f"請求超時: {house_url}")
        return None
    except requests.exceptions.ConnectionError:
        logging.error(f"連接錯誤: {house_url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"無法訪問 {house_url}，錯誤: {e}")
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

            # 檢查必填資料
            if name == "未知名稱" or price is None or size is None:
                logging.warning(f"資料不完整，跳過房屋 ID: {house_id}")
                return None

            # 回傳房屋資料
            house_data = {
                'ID': house_id,
                'Name': name,
                'Price': price,
                'Size': size,
                'Age': age,
                'Floor': floor,
                'Location': location,
                'Tags': ','.join(tags) if tags else ""
            }

            sleep_delay(rp)

            return house_data
        else:
            logging.warning(f"無法從 {house_url} 提取 JavaScript 資料")
            return None
    except Exception as e:
        logging.error(f"解析 {house_url} 時出錯: {e}")
        return None

# 並行抓取所有房屋詳細資訊
def fetch_all_house_data(house_links, rp, max_workers=5):
    house_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_house_data, url, rp): url for url in house_links}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    house_data.append(data)
            except Exception as e:
                logging.error(f"抓取 {url} 的資料時出錯: {e}")
    logging.info(f"成功提取 {len(house_data)} 筆房屋詳細資料。")
    return house_data

# 更新 CSV 的函數
def update_csv(house_data, filename='House_Rent_Info.csv'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"讀取現有的 CSV 檔案: {file_path}")
        except pd.errors.EmptyDataError:
            logging.warning(f"{filename} 是空檔案，初始化為空 DataFrame")
            df = pd.DataFrame(columns=['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags'])
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Price', 'Size', 'Age', 'Floor', 'Location', 'Tags'])
        logging.info(f"創建新的 CSV 檔案: {file_path}")

    new_df = pd.DataFrame(house_data)

    # 移除資料缺失的記錄（已在抓取時處理，但再檢查一次）
    new_df = new_df[
        (new_df['Name'] != "未知名稱") &
        (new_df['Price'].notnull()) &
        (new_df['Size'].notnull())
    ]

    if not df.empty:
        # 使用 'ID' 作為索引進行合併，避免重複
        df.set_index('ID', inplace=True)
        new_df.set_index('ID', inplace=True)

        # 更新價格等資訊，如果 ID 存在且價格不同，則更新
        df.update(new_df)

        # 找出新的資料（ID 不在現有的 DataFrame 中）
        new_entries = new_df[~new_df.index.isin(df.index)]

        # 合併現有資料和新的資料
        df = pd.concat([df, new_entries])

        # 重設索引
        df.reset_index(inplace=True)
    else:
        df = new_df

    # 保存資料到檔案
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    logging.info(f"數據已保存至 {file_path}")

# 主程式
def main():
    BASE_URL = "https://www.rakuya.com.tw/sell/result?city=15&sort=11&page={}"
    TOTAL_PAGES = 1  # 爬取頁數
    DATA_FILENAME = 'House_Rent_Info.csv'

    # 抓取並解析 robots.txt
    robots_txt = fetch_robots_txt("https://www.rakuya.com.tw")
    rp = get_robot_parser("https://www.rakuya.com.tw", robots_txt)

    # 抓取房屋連結
    house_links = fetch_house_links(BASE_URL, rp, max_pages=TOTAL_PAGES)

    # 過濾已存在的連結
    house_links = filter_existing_links(house_links, filename=DATA_FILENAME)

    if house_links:
        # 抓取所有房屋詳細資料
        house_data = fetch_all_house_data(house_links, rp, max_workers=5)

        if house_data:
            # 更新 CSV 檔案
            update_csv(house_data, filename=DATA_FILENAME)
    else:
        logging.info("沒有新的連結需要抓取。")

if __name__ == "__main__":
    main()
