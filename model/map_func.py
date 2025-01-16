import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import requests
import asyncio
import aiohttp
from functools import lru_cache

# 台灣範圍
TAIWAN_BOUNDS = [[20.0, 119.0], [25.5, 124.5]]  # [南緯, 東經] 和 [北緯, 西經]

# 地理編碼設定
geolocator = Nominatim(user_agent="my_geo_app")
SEARCH_RADIUS = 1000  # 搜尋範圍 (以公尺為單位)

PLACE_CATEGORIES = [
    "amenity",            # 便利設施 (包含餐廳、咖啡廳等)
    "shop",               # 商店
    "leisure",            # 娛樂設施
    "tourism",            # 旅遊相關 (包含景點)
    "education",          # 教育設施
    "healthcare",         # 醫療設施
    "public_transport",   # 公共交通 (如捷運站、公車站)
    "restaurant",         # 餐廳
    "fast_food",          # 速食店
    "cafe",               # 咖啡廳
    "tram_stop",          # 輕軌站
    "subway_entrance",    # 捷運入口
    "bus_stop",           # 公車站
    "train_station",      # 火車站
    "parking",            # 停車場
]

@lru_cache(maxsize=1000)
def get_coordinates_cached(address):
    """
    透過地址獲取經緯度並使用緩存。
    """
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        raise Exception(f"無法找到該地址的座標：{address}")

async def fetch_overpass(session, query):
    overpass_url = "http://overpass-api.de/api/interpreter"
    async with session.get(overpass_url, params={'data': query}) as response:
        if response.status == 200:
            return await response.json()
        else:
            response.raise_for_status()

async def search_osm_data_async(lat, lng, categories, radius=SEARCH_RADIUS):
    """
    使用非同步請求來搜尋多個類別的地點。
    """
    category_filters = ''.join([f'node["{cat}"](around:{radius},{lat},{lng});' for cat in categories])
    query = f"""
    [out:json];
    (
      {category_filters}
    );
    out center;
    """
    async with aiohttp.ClientSession() as session:
        data = await fetch_overpass(session, query)
        elements = data.get("elements", [])
        results = {category: [] for category in categories}
        for e in elements:
            tags = e.get('tags', {})
            for category in categories:
                if category in tags:
                    results[category].append((e['lat'], e['lon']))
        return results

def search_osm_data(lat, lng, categories, radius=SEARCH_RADIUS):
    return asyncio.run(search_osm_data_async(lat, lng, categories, radius))

def generate_map(address, map_width='100%', map_height='400px', zoom_start=16):
    """
    根據地址生成 Folium 地圖，並限定在台灣範圍內，同時統計附近設施數量。
    """
    try:
        lat, lng = get_coordinates_cached(address)
        # 檢查是否在台灣範圍內
        if not (TAIWAN_BOUNDS[0][0] <= lat <= TAIWAN_BOUNDS[1][0] and TAIWAN_BOUNDS[0][1] <= lng <= TAIWAN_BOUNDS[1][1]):
            raise Exception("地址不在台灣範圍內。")
        
        folium_map = folium.Map(
            location=[lat, lng], 
            zoom_start=zoom_start, 
            width=map_width, 
            height=map_height,
            min_zoom=10,  # 最小縮放級別，限制只能縮放到台灣範圍內
            max_bounds=True,  # 限制地圖在台灣範圍內可滑動
            max_lat=TAIWAN_BOUNDS[1][0],  # 設定最大緯度 (北緯)
            min_lat=TAIWAN_BOUNDS[0][0],  # 設定最小緯度 (南緯)
            max_lon=TAIWAN_BOUNDS[1][1],  # 設定最大經度 (東經)
            min_lon=TAIWAN_BOUNDS[0][1],  # 設定最小經度 (西經)
        )

        folium.Marker(
            location=[lat, lng], 
            popup=address, 
            icon=folium.Icon(color="red")
        ).add_to(folium_map)

        # 設施數量統計字典
        place_count = {category: 0 for category in PLACE_CATEGORIES}

        # 使用單一 MarkerCluster
        cluster = MarkerCluster().add_to(folium_map)

        # 一次性查詢所有類別的設施
        osm_results = search_osm_data(lat, lng, PLACE_CATEGORIES)

        for category, locations in osm_results.items():
            place_count[category] = len(locations)
            for loc in locations:
                folium.Marker(
                    location=loc, 
                    popup=category, 
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(cluster)

        # 返回生成的地圖和統計結果
        return folium_map._repr_html_(), place_count

    except Exception as e:
        return f"<p>地圖生成失敗：{str(e)}</p>", {}
