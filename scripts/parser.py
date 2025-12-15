import pandas as pd
import requests
import re
import time
from tqdm import tqdm

# TODO: перенести в env
YANDEX_API_KEY = ''
INPUT_DEALS = 'data/Сделки_2025-11-25.xlsx'
INPUT_FLATS = 'data/Проектные_данные_bnMAPpro.xlsx'

OUTPUT_ADDR = 'cache/cache_addresses.csv'
OUTPUT_FEAT = 'cache/cache_features.csv'

def clean_yandex_address(addr):
    """Очистка адреса для лучшего геокодирования."""
    if pd.isna(addr) or addr == '':
        return ''

    res = str(addr)
    res = re.sub(r'\s*\(.*?\)', '', res)

    replacements = {
        r'\bпос\.': 'поселение', r'\bдер\.': 'деревня', r'\bд\.': 'дом',
        r'\bк\.': 'корпус', r'\bстр\.': 'строение', r'\bул\.': 'улица',
        r'\bмкр\.': 'микрорайон', r'\bуч\.': 'участок', r'\bп\.': 'поселение',
        r'\bтер\.': 'территория', r'\bкв-л': 'квартал', r'\bз/у': 'земельный участок',
        r'\bвблизи': '',
    }
    for pattern, repl in replacements.items():
        res = re.sub(pattern, repl + ' ', res, flags=re.IGNORECASE)

    if 'москва' not in res.lower():
        res = 'Москва, ' + res

    res = re.sub(r'\s+', ' ', res).replace('.,', ',').strip(', ')
    return res

def fetch_coordinates_yandex(address_list):
    """Парсинг координат через Yandex Maps API."""
    coords_map = {}
    base_url = "https://geocode-maps.yandex.ru/1.x"

    print(f"Начинаем геокодирование {len(address_list)} адресов...")

    for addr in tqdm(address_list):
        if not addr:
            continue

        params = {"apikey": YANDEX_API_KEY, "geocode": addr, "format": "json", "results": 1}
        try:
            resp = requests.get(base_url, params=params)
            data = resp.json()
            feature = data['response']['GeoObjectCollection']['featureMember']

            if feature:
                pos = feature[0]['GeoObject']['Point']['pos']
                lon, lat = map(float, pos.split(' '))
                coords_map[addr] = (lat, lon)
            else:
                coords_map[addr] = (None, None)
        except Exception:
            coords_map[addr] = (None, None)

        time.sleep(0.05)

    return coords_map

def get_osm_data():
    """
    Скачивание инфраструктуры OSM частями (чтобы избежать TimeOut)
    """
    print("Скачивание данных OSM (Overpass API)...")
    url = "http://overpass-api.de/api/interpreter"
    bbox = "(55.1,36.8,56.1,38.2)"

    # Разбиваем один большой запрос на 3 части
    # Дороги
    query_roads = f"""
    [out:json][timeout:180];
    (
      way["name"~"МКАД"]{bbox};
      way["ref"~"A-113|А-113"]{bbox};
      way["name"~"Калужское шоссе"]{bbox};
      way["name"~"Киевское шоссе"]{bbox};
      way["name"~"Варшавское шоссе"]{bbox};
      way["name"~"Боровское шоссе"]{bbox};
      way["name"~"Солнцево.*Бутово|СБВ"]{bbox};

      // Метро и ЖД точками
      node["station"="subway"]{bbox};
      node["railway"="station"]{bbox};
      node["railway"="halt"]{bbox};
    );
    out geom tags; 
    """

    # Социалка и Торговля
    query_social = f"""
    [out:json][timeout:180];
    (
      node["amenity"="school"]{bbox}; way["amenity"="school"]{bbox};
      node["amenity"="kindergarten"]{bbox}; way["amenity"="kindergarten"]{bbox};
      node["amenity"~"clinic|hospital"]{bbox}; way["amenity"~"clinic|hospital"]{bbox};
      node["amenity"="police"]{bbox}; way["amenity"="police"]{bbox};
      node["amenity"="prison"]{bbox}; way["amenity"="prison"]{bbox};

      node["shop"~"supermarket|convenience"]{bbox}; way["shop"~"supermarket|convenience"]{bbox};
      node["shop"~"mall|department_store"]{bbox}; way["shop"~"mall|department_store"]{bbox};
    );
    out center tags;
    """

    # Природа, ЛЭП и Кладбища
    query_nature = f"""
    [out:json][timeout:180];
    (
      node["leisure"~"park|garden"]{bbox}; way["leisure"~"park|garden"]{bbox};
      node["natural"="wood"]{bbox}; way["natural"="wood"]{bbox};
      node["landuse"="forest"]{bbox}; way["landuse"="forest"]{bbox};

      node["landuse"="cemetery"]{bbox}; way["landuse"="cemetery"]{bbox};
      node["amenity"="grave_yard"]{bbox}; way["amenity"="grave_yard"]{bbox};

      node["power"="tower"]{bbox}; way["power"="line"]{bbox};
    );
    out center tags;
    """

    queries = [
        ("Дороги и транспорт", query_roads),
        ("Социалка и торговля", query_social),
        ("Природа и ЛЭП", query_nature)
    ]

    all_elements = []

    # ВЫПОЛНЯЕМ ЗАПРОСЫ ПО ОЧЕРЕДИ
    for q_name, q_body in queries:
        print(f"--> Запрос: {q_name}...")
        data = requests.get(url, params={'data': q_body}).json()

        elements = data.get('elements', [])
        all_elements.extend(elements)
        print(f"    Получено объектов: {len(elements)}")

        time.sleep(1)

    print(f"Всего загружено объектов: {len(all_elements)}")

    # 3. ОБРАБОТКА ДАННЫХ
    mkad_blacklist = [
        (55.5638032, 37.4679931),
        (55.5642968, 37.4682043), (55.5641857, 37.4682018),
        (55.5640997, 37.468174),  (55.5639867, 37.4681316),
        (55.5639867, 37.4681316), (55.5635395, 37.4677567),
        (55.5634811, 37.4679668), (55.5632503, 37.4674572),
        (55.5631798, 37.4676758), (55.5629658, 37.4671209),
        (55.5628985, 37.4673459), (55.5613891, 37.4649399),
        (55.5612917, 37.4651499), (55.561239, 37.4647156),
        (55.5611695, 37.4646118), (55.5610212, 37.4647628),
        (55.5608693, 37.4645289), (55.5606663, 37.4639859),
        (55.5605949, 37.4641065), (55.5603964, 37.4637035),
        (55.5603494, 37.4637946), (55.560122, 37.4634807),
    ]

    ckad_blacklist = [
        (55.6688949, 37.4807735), (55.7318204, 37.5752147),
        (55.6688951, 37.4808889), (55.7318032, 37.575187),
        (55.668858, 37.4808892), (55.6688578, 37.4807735),
        (55.7318425, 37.5751718), (55.7318282, 37.5751997),
        (55.7318008, 37.5751042), (55.7317854, 37.5750791),
        (55.7317802, 37.5750893), (55.7317711, 37.5751068),
        (55.7317761, 37.5751427), (55.7317633, 37.575122),
        (55.7317685, 37.5751303)
    ]

    objects = []

    for el in all_elements:
        tags = el.get('tags', {})
        name = tags.get('name', 'Без названия')
        name_lower = name.lower()
        obj_type = None

        ref = str(tags.get('ref', '')).lower()


        # --- КЛАССИФИКАЦИЯ ---
        if 'a-113' in ref or 'а-113' in ref:
            obj_type = 'ckad'

        if 'мкад' in name_lower: obj_type = 'mkad'
        elif 'калужское шоссе' in name_lower: obj_type = 'kaluzhskoe'
        elif 'киевское шоссе' in name_lower: obj_type = 'kievskoe'
        elif 'варшавское шоссе' in name_lower and 'андреевское' not in name_lower: obj_type = 'varshavskoe'
        elif 'боровское шоссе' in name_lower: obj_type = 'borovskoe'
        elif ('солнцево' in name_lower and 'бутово' in name_lower) or 'сбв' in name_lower: obj_type = 'sbv'

        elif tags.get('station') == 'subway': obj_type = 'metro'
        elif tags.get('railway') == 'station' or tags.get('railway') == 'halt': obj_type = 'train'

        elif tags.get('amenity') == 'school': obj_type = 'school'
        elif tags.get('amenity') == 'kindergarten': obj_type = 'kindergarten'
        elif tags.get('amenity') in ['clinic', 'hospital']: obj_type = 'medical'
        elif tags.get('amenity') == 'police': obj_type = 'police'
        elif tags.get('amenity') == 'prison': obj_type = 'prison'

        elif tags.get('shop') in ['supermarket', 'convenience']: obj_type = 'grocery'
        elif tags.get('shop') in ['mall', 'department_store']: obj_type = 'mall'

        elif tags.get('leisure') in ['park', 'garden']: obj_type = 'park'
        elif tags.get('natural') == 'wood' or tags.get('landuse') == 'forest': obj_type = 'forest'
        elif tags.get('landuse') == 'cemetery' or tags.get('amenity') == 'grave_yard': obj_type = 'cemetery'

        elif tags.get('power') in ['line', 'tower']: obj_type = 'power_line'

        if not obj_type:
            continue

        # --- ИЗВЛЕЧЕНИЕ КООРДИНАТ ---

        points_to_add = []

        # У объекта есть "geometry" (это линия дороги из первого запроса)
        if 'geometry' in el:
            # Берем КАЖДУЮ точку линии
            for p in el['geometry']:
                points_to_add.append((p['lat'], p['lon']))
        # У объекта есть "lat/lon" или "center" (точка)
        else:
            lat = el.get('lat') or el.get('center', {}).get('lat')
            lon = el.get('lon') or el.get('center', {}).get('lon')
            if lat and lon:
                points_to_add.append((lat, lon))

        # Фильтр МКАДа
        for lat, lon in points_to_add:
            if obj_type == 'mkad' or obj_type == 'ckad':
                is_blacklisted = False
                for bad_lat, bad_lon in mkad_blacklist+ckad_blacklist:
                    if abs(lat - bad_lat) < 0.000001 and abs(lon - bad_lon) < 0.000001:
                        is_blacklisted = True
                        break
                if is_blacklisted:
                    continue

            objects.append({
                'lat': lat,
                'lon': lon,
                'type': obj_type,
                'name': name
            })

    # 4. ДОБАВЛЕНИЕ АЭРОПОРТОВ
    airports_list = [
        {'name': 'Шереметьево (SVO)', 'lat': 55.972642, 'lon': 37.414589},
        {'name': 'Домодедово (DME)', 'lat': 55.410307, 'lon': 37.902451},
        {'name': 'Внуково (VKO)', 'lat': 55.591531, 'lon': 37.261486},
        {'name': 'Жуковский (ZIA)', 'lat': 55.561732, 'lon': 38.118255},
        {'name': 'Остафьево (OSF)', 'lat': 55.509722, 'lon': 37.505556},
        {'name': 'Чкаловский (CKL)', 'lat': 55.878333, 'lon': 38.058333},
        {'name': 'Мячково', 'lat': 55.558333, 'lon': 37.983333},
        {'name': 'Кубинка', 'lat': 55.611667, 'lon': 36.650000},
        {'name': 'Северка', 'lat': 55.208333, 'lon': 38.675000},
        {'name': 'Финам (Большое Грызлово)', 'lat': 54.785000, 'lon': 37.648333},
        {'name': 'Ватулино', 'lat': 55.661667, 'lon': 36.141667},
        {'name': 'Черное (Марз)', 'lat': 55.760000, 'lon': 38.061667},
        {'name': 'Коробчеево (Аэроград)', 'lat': 55.093333, 'lon': 38.831667},
        {'name': 'Дракино', 'lat': 54.873333, 'lon': 37.271667}
    ]

    for air in airports_list:
        objects.append({'lat': air['lat'], 'lon': air['lon'], 'type': 'airport', 'name': air['name']})

    return pd.DataFrame(objects)


# 1. Загрузка исходных данных для получения списка адресов
print("Чтение исходных файлов...")
df1 = pd.read_excel(INPUT_DEALS)
df2 = pd.read_excel(INPUT_FLATS)


# Извлечение уникальных адресов
addrs1 = df1['Адрес корпуса'].apply(clean_yandex_address).unique()
addrs2 = df2['Адрес'].apply(clean_yandex_address).unique()
unique_addresses = list(set(addrs1) | set(addrs2))
unique_addresses = [x for x in unique_addresses if x]

# 2. Геокодирование и сохранение таблицы 1 (Адрес -> Координаты)
coords_map = fetch_coordinates_yandex(unique_addresses)

df_addr_cache = pd.DataFrame([
    {'clean_address': k, 'lat': v[0], 'lon': v[1]}
    for k, v in coords_map.items() if v[0] is not None
])
df_addr_cache.to_csv(OUTPUT_ADDR, index=False)
print(f"Справочник адресов сохранен: {OUTPUT_ADDR}")


# 3. Подготовка уникальных координат для расчета метрик
# Округляем до 5 знаков (~1 метр), чтобы сгруппировать точки
unique_coords = df_addr_cache[['lat', 'lon']].round(5).drop_duplicates().values


# 4. Скачивание OSM данных
osm_df = get_osm_data()
osm_df.to_csv(OUTPUT_FEAT, index=False)
print(f"База инфраструктуры сохранена: {OUTPUT_FEAT}")