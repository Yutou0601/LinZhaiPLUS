##############################################
# model/Crawler_Info.py
##############################################
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import logging

# ===== 日誌設定 =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start = datetime.now()

page_url = 'https://www.rakuya.com.tw/rent/rent_search?search=city&city=99&upd=1&page='
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'House_Rent_Info.csv')    # <--- 產生在 data 資料夾下
house_data_df = pd.DataFrame()

# 讀取現有的 CSV 並取得已存在的 Name 和 ID 集合
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    if 'ID' in existing_df.columns:
        existing_names = set(existing_df['Name'])
        existing_ids = set(existing_df['ID'])
        max_existing_id = existing_df['ID'].max()
    else:
        existing_df = pd.DataFrame()
        existing_names = set()
        existing_ids = set()
        max_existing_id = 0
else:
    existing_df = pd.DataFrame()
    existing_names = set()
    existing_ids = set()
    max_existing_id = 0

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://www.rakuya.com.tw'
}

@dataclass
class HouseData:
    ID: int  # 新增 ID 欄位
    Name: str
    Price: str
    Size: str
    Age: str
    Floors: str
    Bedroom: str
    City: str
    Location: str
    HouseType: str
    Pattern: str
    Tags: list[str] = field(default_factory=list)
    Environment: list[str] = field(default_factory=list)
    Url: str = ''
    Image_name: str = ''  # 存放圖片的檔案名稱或路徑

    def to_dict(self):
        data = asdict(self)
        data['Tags'] = ', '.join(self.Tags)
        data['Environment'] = ', '.join(self.Environment)  # 儲存為逗號分隔的字串
        return data

    def to_df(self):
        return pd.DataFrame([self.to_dict()])

def find_max_pages():
    resp = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    pages_text = soup.find('p', class_='pages')
    if pages_text:
        pages_data = re.findall(r'\d+', pages_text.text)
        return int(pages_data[1]) if len(pages_data) > 1 else 1
    return 1

def crawler_url_set(url, page=1):
    link_list = []
    title_list = []

    for i in range(1, page+1):
        current_page_url = url + str(i)
        resp = requests.get(current_page_url, headers=headers)
        if resp.status_code != 200:
            logging.warning(f"無法訪問頁面: {current_page_url}，狀態碼: {resp.status_code}")
            continue
        soup = BeautifulSoup(resp.text, 'html.parser')
        items = soup.select('h6 > a[href*="/rent/rent_item"]')
        for item in items:
            link = item.get('href')
            if not link:
                continue
            title = item.get_text(strip=True)
            link_list.append(link)
            title_list.append(title)

    return {
        "Title": title_list,
        "Link": link_list
    }

def download_image(image_url, listing_title):
    """
    下載圖片並儲存到 static/Images 資料夾。
    回傳儲存的圖片檔案名稱。
    """
    # 創建 static/Images 資料夾（如果尚未存在）
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'Images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        logging.info(f"已創建資料夾: {images_dir}")

    # 生成圖片檔案名稱，避免重複
    # 使用哈希值來確保唯一性
    image_ext = os.path.splitext(image_url)[1].split('?')[0]  # 取得副檔名
    hash_object = hashlib.md5(image_url.encode())
    image_name = f"{hash_object.hexdigest()}{image_ext}"
    image_path = os.path.join(images_dir, image_name)

    # 檢查圖片是否已存在，避免重複下載
    if os.path.exists(image_path):
        logging.info(f"圖片已存在，跳過下載: {image_name}")
        return image_name

    # 下載圖片
    try:
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

    return ''  # 若下載失敗，返回空字串

def crawler_house_info(raw_title, link, overwrite_id):
    global existing_names, house_data_df, existing_ids
    if raw_title in existing_names:
        logging.info(f"已存在的房源，跳過: {raw_title}")
        return
    resp = requests.get(link, headers=headers)
    if resp.status_code != 200:
        logging.warning(f"無法訪問房源頁面: {link}，狀態碼: {resp.status_code}")
        return
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 嘗試抓資料
    content = soup.find('script', string=lambda s: s and 'window.tmpDataLayer' in s)
    if not content:
        logging.warning(f"找不到資料層腳本，跳過: {raw_title}")
        return
    data = content.string.strip()

    # title
    item_name = raw_title
    # 用正則取 price, age, size...
    price_match = re.search(r'"price":(\d+)', data)
    price = price_match.group(1) if price_match else '0'

    age_match = re.search(r'"age":(\d+\.?\d*)', data)
    age = age_match.group(1) if age_match else '0'

    size_match = re.search(r'"item_variant":(\d+\.?\d*)', data)
    size = size_match.group(1) if size_match else '0'

    floor_match = re.search(r'"object_floor":(\d+)', data)
    floor = floor_match.group(1) if floor_match else '0'

    bedroom_match = re.search(r'"bedrooms":(\d+)', data)
    bedroom = bedroom_match.group(1) if bedroom_match else '0'

    city_match = re.search(r'"item_category":"(.*?)"', data)
    city = city_match.group(1) if city_match else ''

    htype_match = re.search(r'"item_category5":"(.*?)"', data)
    htype = htype_match.group(1) if htype_match else ''

    tags_match = re.search(r'"object_tag":"(.*?)"', data)
    tags = tags_match.group(1).split(',') if tags_match else []

    # address
    address_tag = soup.find('h1', class_='txt__address')
    location = address_tag.get_text(strip=True) if address_tag else ''

    # pattern, environment
    pattern = ''
    environment = []
    li_elems = soup.find_all('li')
    for li in li_elems:
        label = li.find('span', class_='list__label')
        if label and "格局" in label.text:
            pattern = li.find('span', class_='list__content').get_text(strip=True)
        elif label and "物件環境" in label.text:
            b_tags = li.find('span', class_='list__content').find_all('b')
            environment = [b.get_text(strip=True) for b in b_tags]

    # 提取圖片 URL
    image_url = ''
    image_tag = soup.find('img', class_='main-image')  # 假設圖片有 class='main-image'
    if image_tag and image_tag.get('src'):
        image_url = image_tag.get('src')
    else:
        # 嘗試其他方式提取圖片
        image_container = soup.find('div', class_='image-container')
        if image_container:
            image_tag = image_container.find('img')
            if image_tag and image_tag.get('src'):
                image_url = image_tag.get('src')

    # 下載圖片並儲存
    image_name = ''
    if image_url:
        image_name = download_image(image_url, raw_title)

    # 封裝
    house_obj = HouseData(
        ID=overwrite_id,  # 填入覆蓋的 ID
        Name=item_name,
        Price=price,
        Size=size,
        Age=age,
        Floors=floor,
        Bedroom=bedroom,
        City=city,
        Location=location,
        HouseType=htype,
        Pattern=pattern,
        Tags=tags,
        Environment=environment,
        Url=link,
        Image_name=image_name  # 記錄下載的圖片名稱
    )
    house_data_df = pd.concat([house_data_df, house_obj.to_df()], ignore_index=True)
    logging.info(f"已抓取房源資料: {item_name}, 覆蓋 ID: {overwrite_id}")

if __name__ == '__main__':
    page_count = 10  # 設定要爬取的頁數
    contain = crawler_url_set(page_url, page_count)
    df_url = pd.DataFrame({"Title": contain['Title'], "Link": contain['Link']})
    
    for idx, (title, link) in enumerate(zip(df_url['Title'], df_url['Link']), start=1):
        # 計算要覆蓋的 ID，從1開始循環
        if max_existing_id >= 300:
            overwrite_id = (idx - 1) % 300 + 1
        else:
            overwrite_id = max_existing_id + idx
        crawler_house_info(title, link, overwrite_id)
    
    # 最終儲存 CSV
    end = datetime.now()
    if not house_data_df.empty:
        if not existing_df.empty:
            if 'ID' in existing_df.columns:
                # 將新的資料覆蓋到 existing_df 的對應 ID
                combined_df = existing_df.set_index('ID').copy()
                new_df = house_data_df.set_index('ID')
                combined_df.update(new_df)
                combined_df.reset_index(inplace=True)
            else:
                combined_df = house_data_df
        else:
            combined_df = house_data_df
        # ★★ 新增行: 在 CSV 多加一欄 CP_value=0 ★★
        combined_df['CP_value'] = 0

        combined_df.to_csv(csv_path, index=False)
        print(f"[完成] 已覆蓋 {len(house_data_df)} 筆資料至 {csv_path}")
        logging.info(f"[完成] 已覆蓋 {len(house_data_df)} 筆資料至 {csv_path}")
    else:
        print("[完成] 無新資料需寫入")
        logging.info("[完成] 無新資料需寫入")

    print(f"Time Spent: {end - start} ")
    logging.info(f"Time Spent: {end - start} ")
