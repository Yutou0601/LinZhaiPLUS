import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import requests

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

def get_coordinates(address):
    """
    透過地址獲取經緯度
    """
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        raise Exception(f"無法找到該地址的座標：{address}")

def search_osm_data(lat, lng, category, radius=SEARCH_RADIUS):
    """
    使用 OSM API 搜尋附近的地點，並限制範圍在台灣區域內
    """
    # 限制搜尋範圍，確保不會超出台灣範圍
    if not (TAIWAN_BOUNDS[0][0] <= lat <= TAIWAN_BOUNDS[1][0] and TAIWAN_BOUNDS[0][1] <= lng <= TAIWAN_BOUNDS[1][1]):
        return []

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node[{category}](around:{radius},{lat},{lng});
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code == 200:
        data = response.json()
        elements = data.get("elements", [])
        return [(e['lat'], e['lon']) for e in elements if 'lat' in e and 'lon' in e]
    return []

def generate_map(address, map_width='100%', map_height='400px', zoom_start=16):
    """
    根據地址生成 Folium 地圖，並限定在台灣範圍內，同時統計附近設施數量
    """
    try:
        lat, lng = get_coordinates(address)
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

        # 生成設施標記並統計
        for category in PLACE_CATEGORIES:
            locations = search_osm_data(lat, lng, category)
            cluster = MarkerCluster(name=category).add_to(folium_map)
            place_count[category] = len(locations)  # 計算每個類別的設施數量
            for loc in locations:
                folium.Marker(
                    location=loc, 
                    popup=category, 
                    icon=folium.Icon(color="blue")
                ).add_to(cluster)

        # 返回生成的地圖和統計結果
        return folium_map._repr_html_(), place_count

    except Exception as e:
        return f"<p>地圖生成失敗：{str(e)}</p>", {}

