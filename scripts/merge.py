import pandas as pd
import numpy as np
import re
from tqdm import tqdm

INPUT_DEALS = 'data/Сделки_2025-11-25.xlsx'
INPUT_FLATS = 'data/Проектные_данные_bnMAPpro.xlsx'

CACHE_ADDR = 'cache/cache_addresses.csv'
CACHE_INFRA = 'cache/cache_features.csv'

OUTPUT_DEALS = 'data/deals.csv'
OUTPUT_FLATS = 'data/flats.csv'

OUTPUT_ERR_DEALS = 'cache/deleted_deals.csv'
OUTPUT_ERR_FLATS = 'cache/deleted_flats.csv'

KREMLIN_LAT = 55.75222
KREMLIN_LON = 37.61556

MANUAL_COORDS = {
    'Москва, Лесные Поляны 5-я улица , земельный участок 20А, корпус 1': (55.585210, 37.463180),
    'Москва, Лесные Поляны 5-я улица , земельный участок 20А, корпус 2': (55.586549, 37.463367),

    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.3': (55.556304, 37.515632),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.1, корпус 18.1.4': (55.556716, 37.514302),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.6': (55.556120, 37.517311),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.1, корпус 18.1.2': (55.557220, 37.513691),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.5': (55.556549, 37.517305),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.1, корпус 18.1.3': (55.557098, 37.514338),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.1, корпус 18.1.1': (55.557424, 37.513215),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.2': (55.556752, 37.515784),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.1': (55.556487, 37.514787),
    'Москва, Поляны улица , земельный участок 50Д, дом 18.2, корпус 18.2.4': (55.556487, 37.514787),

    'Москва, Лаптева улица , земельный участок 2, корпус 3': (55.597047, 37.367526),
    'Москва, Лаптева улица , земельный участок 4, корпус 3': (55.600699, 37.366493),
    'Москва, Лаптева улица , земельный участок 4, корпус 4': (55.600699, 37.366493),

    'Москва, Сосенское поселение , квартал 26, земельный участок 3/3, корпус 11.2.2': (55.595105, 37.437657),

    'Москва, Газопровод поселение , корпус Aurora': (55.590329, 37.483488),
    'Москва, Газопровод поселение , корпус Essense': (55.589631, 37.483459),

    'Москва, СНТ Ветеран-1 территория , корпус 1': (55.492605, 37.319475),

    'Москва, Александры Монаховой улица , земельный участок 87, корпус 3.7.2': (55.550005, 37.488251)
}

def apply_manual_fixes(df, manual_dict):
    """
    Применяет ручные координаты к датафрейму по совпадению clean_address.
    """
    count = 0

    for addr, (lat, lon) in manual_dict.items():
        mask = df['clean_address'] == addr

        if mask.any():
            df.loc[mask, 'lat'] = lat
            df.loc[mask, 'lon'] = lon
            count += mask.sum()

    print(f"Ручные координаты применены к {count} строкам.")
    return df

def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """Расстояние от точки до массива точек (км)"""
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_array), np.radians(lon2_array)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def filter_bad_coordinates(df, dataset_name):
    """
    Удаляет строки, где координаты:
    1. Отсутствуют.
    2. Попадают в центр Москвы (ошибка геокодирования).
    3. Вылетают за пределы разумного региона (по широте и долготе).
    Возвращает: (хороший_df, плохой_df)
    """
    print(f"[{dataset_name}] Проверка координат на адекватность...")
    initial_len = len(df)

    mask_nan = df['lat'].isna() | df['lon'].isna()

    # 2. Проверка расстояния до Кремля
    dist_to_center = haversine_vectorized(df['lat'], df['lon'], KREMLIN_LAT, KREMLIN_LON)
    mask_in_center = dist_to_center < 14.0

    # 3. Проверка границ (Bounding Box для Новой Москвы)
    mask_lat_bad = (df['lat'] > 55.72) | (df['lat'] < 55.05)
    mask_lon_bad = (df['lon'] > 37.75) | (df['lon'] < 36.70)

    mask_bounds = mask_lat_bad | mask_lon_bad

    mask_bad = mask_nan | mask_in_center | mask_bounds

    bad_df = df[mask_bad].copy()
    good_df = df[~mask_bad].copy()

    bad_df['error_reason'] = 'Unknown'

    bad_df.loc[mask_bounds & ~mask_nan, 'error_reason'] = 'Вне границ Новой Москвы (lat/lon)'
    bad_df.loc[mask_in_center & ~mask_nan, 'error_reason'] = 'Попал в Старую Москву (ошибка адреса)'
    bad_df.loc[mask_nan, 'error_reason'] = 'Координаты не найдены'

    print(f"[{dataset_name}] Было: {initial_len}, Удалено: {len(bad_df)}, Осталось: {len(good_df)}")

    return good_df, bad_df

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

def find_nearest_simple(target_df, infra_df, infra_type, prefix):
    print(f"--- Расчет: {prefix} ({infra_type}) ---")
    subset = infra_df[infra_df['type'] == infra_type].copy()

    infra_lats = subset['lat'].values
    infra_lons = subset['lon'].values
    infra_names = subset['name'].values

    result_dists = []
    result_names = []

    for idx, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
        lat, lon = row['lat'], row['lon']

        dists = haversine_vectorized(lat, lon, infra_lats, infra_lons)
        min_idx = np.argmin(dists)

        result_dists.append(round(dists[min_idx], 3))
        result_names.append(infra_names[min_idx])

    target_df[f'{prefix}_dist_km'] = result_dists
    target_df[f'{prefix}_name'] = result_names
    return target_df

def count_nearby_simple(target_df, infra_df, infra_type, col_name, radius_km=0.5):
    """
    Считает количество объектов заданного типа в радиусе radius_km.
    """
    print(f"--- Подсчет: {col_name} ({infra_type} в радиусе {radius_km*1000}м) ---")

    subset = infra_df[infra_df['type'] == infra_type].copy()

    infra_lats = subset['lat'].values
    infra_lons = subset['lon'].values

    result_counts = []

    for idx, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
        lat, lon = row['lat'], row['lon']

        dists = haversine_vectorized(lat, lon, infra_lats, infra_lons)

        count = np.sum(dists <= radius_km)
        result_counts.append(count)

    target_df[col_name] = result_counts
    return target_df

def process_pipeline(df, addr_col, cache_addr, infra_db, dataset_name):
    print(f"Мердж координат для {dataset_name}...")
    df['clean_address'] = df[addr_col].apply(clean_yandex_address)
    df = df.merge(cache_addr, on='clean_address', how='left')

    df = apply_manual_fixes(df, MANUAL_COORDS)

    good_df, bad_df = filter_bad_coordinates(df, dataset_name)

    search_map = {
        'metro': 'metro',
        'train': 'train_station',
        'mkad': 'mkad',
        'kaluzhskoe': 'kaluzhskoe',
        'kievskoe': 'kievskoe',
        'varshavskoe': 'varshavskoe',
        'borovskoe': 'borovskoe',
        'school': 'school',
        'kindergarten': 'kindergarten',
        'medical': 'medical',
        'police': 'police',
        'prison': 'prison',
        'mall': 'mall',
        'grocery': 'grocery',
        'park': 'park',
        'forest': 'forest',
        'cemetery': 'cemetery',
        'airport': 'airport',
        'power_line': 'power_line'
    }

    for osm_type, col_prefix in search_map.items():
        good_df = find_nearest_simple(good_df, infra_db, osm_type, col_prefix)

    good_df = count_nearby_simple(
        target_df=good_df,
        infra_df=infra_db,
        infra_type='grocery',
        col_name='groceries_count_500m',
        radius_km=0.5
    )

    return good_df, bad_df


def merge_flats_to_deals(flats_df, deals_df):
    """
    Объединяет данные из flats в deals по ID корпуса
    """
    print("Начало объединения flats в deals...")

    cols = flats_df.columns
    new_cols = []
    for col in cols:
        if col == 'Этожей до':
            new_cols.append('Этажей до')
        else:
            new_cols.append(col)
    flats_df.columns = new_cols

    flats_agg = flats_df.groupby('ID корпуса')[['Этажей от', 'Этажей до']].min().reset_index()
    deals_df = deals_df.merge(flats_agg, on='ID корпуса', how='left')

    flats_agg = flats_df.groupby('ID корпуса')['Плановая дата РВЭ'].min().reset_index()
    deals_df = deals_df.merge(flats_agg, on='ID корпуса', how='left')

    flats_agg = flats_df.groupby('ID корпуса')['Стадия строительства'].min().reset_index()
    deals_df = deals_df.merge(flats_agg, on='ID корпуса', how='left')

    flats_df['Общая проектная площадь'] = pd.to_numeric(
        flats_df['Общая проектная площадь'], errors='coerce'
    )

    flats_agg = flats_df.groupby('ID корпуса')['Общая проектная площадь'].sum().reset_index()
    deals_df = deals_df.merge(flats_agg, on='ID корпуса', how='left')
    deals_df['Общая проектная площадь'] = pd.to_numeric(
        deals_df['Общая проектная площадь'], errors='coerce'
    ).fillna(0)

    deals_df['Ипотека'] = deals_df['Ипотека'].apply(
        lambda x: 1 if str(x).strip().lower() == 'ипотека' else 0
    )

    deals_df['Суммарная площадь сделок'] = pd.to_numeric(
        deals_df['Суммарная площадь сделок'], errors='coerce'
    ).fillna(0)

    deals_df['Сумма бюджета'] = pd.to_numeric(
        deals_df['Сумма бюджета'], errors='coerce'
    ).fillna(0)

    deals_df['Количество комнат'] = deals_df['Количество комнат'].astype(str).apply(
        lambda x: x if x.isdigit() else 0
    )

    deals_df['Суммарное количество сделок'] = pd.to_numeric(
        deals_df['Суммарное количество сделок'], errors='coerce'
    ).fillna(0)

    deals_df['Цена квадратного метра'] = (
            deals_df['Сумма бюджета'] / deals_df['Суммарная площадь сделок']
    )

    print("Объединение flats в deals завершено")
    return deals_df


def translate_deals(df: pd.DataFrame) -> pd.DataFrame:
    translation_map = {
        'clean_address': 'Чистый адрес',
        'lat': 'Широта',
        'lon': 'Долгота',
        'metro_dist_km': 'Расстояние до станции метро',
        'metro_name': 'Название станции метро',
        'train_station_dist_km': 'Расстояние до ЖД станции',
        'train_station_name': 'Название ЖД станции',
        'mkad_dist_km': 'Расстояние до МКАД',
        'kaluzhskoe_dist_km': 'Расстояние до шоссе Калужское',
        'kievskoe_dist_km': 'Расстояние до шоссе Киевское',
        'varshavskoe_dist_km': 'Расстояние до шоссе Варшавское',
        'borovskoe_dist_km': 'Расстояние до шоссе Боровское',
        'mkad_name': 'Название участка МКАД',
        'school_dist_km': 'Расстояние до школы',
        'school_name': 'Название школы',
        'kindergarten_dist_km': 'Расстояние до детского сада',
        'kindergarten_name': 'Название детского сада',
        'medical_dist_km': 'Расстояние до медицинского учреждения',
        'medical_name': 'Название медицинского учреждения',
        'police_dist_km': 'Расстояние до полиции',
        'police_name': 'Название отделения полиции',
        'prison_dist_km': 'Расстояние до тюрьмы',
        'prison_name': 'Название тюрьмы',
        'mall_dist_km': 'Расстояние до торгового центра',
        'mall_name': 'Название торгового центра',
        'grocery_dist_km': 'Расстояние до продуктового магазина',
        'grocery_name': 'Название продуктового магазина',
        'park_dist_km': 'Расстояние до парка',
        'park_name': 'Название парка',
        'forest_dist_km': 'Расстояние до леса',
        'forest_name': 'Название леса',
        'cemetery_dist_km': 'Расстояние до кладбища',
        'cemetery_name': 'Название кладбища',
        'airport_dist_km': 'Расстояние до аэропорта',
        'airport_name': 'Название аэропорта',
        'power_line_dist_km': 'Расстояние до ЛЭП',
        'power_line_name': 'Название ЛЭП',
        'groceries_count_500m': 'Количество магазинов в радиусе 500 м'
    }

    return df.rename(columns=translation_map)



print("Загрузка данных...")
cache_addr = pd.read_csv(CACHE_ADDR)
infra_db = pd.read_csv(CACHE_INFRA)


# --- FLATS ---
flats = pd.read_excel(INPUT_FLATS)
flats_clean, flats_bad = process_pipeline(flats, 'Адрес', cache_addr, infra_db, 'FLATS')

# --- DEALS ---
deals = pd.read_excel(INPUT_DEALS)
deals_clean, deals_bad = process_pipeline(deals, 'Адрес корпуса', cache_addr, infra_db, 'DEALS')


flats_clean['Минимальная транспортная доступность'] = flats_clean[['metro_dist_km', 'train_station_dist_km']].min(axis=1)
deals_clean['Минимальная транспортная доступность'] = deals_clean[['metro_dist_km', 'train_station_dist_km']].min(axis=1)

deals_clean = merge_flats_to_deals(flats_clean, deals_clean)
deals_bad = merge_flats_to_deals(flats_bad, deals_bad)

deals_clean = translate_deals(deals_clean)
deals_bad = translate_deals(deals_bad)


print("\nСохранение CSV файлов...")

flats_clean.to_csv(OUTPUT_FLATS, index=False, sep=',', encoding='utf-8-sig')
deals_clean.to_csv(OUTPUT_DEALS, index=False, sep=',', encoding='utf-8-sig')

flats_bad.to_csv(OUTPUT_ERR_FLATS, index=False, sep=',', encoding='utf-8-sig')
print(f"Отчет об ошибках квартир: {OUTPUT_ERR_FLATS}")

deals_bad.to_csv(OUTPUT_ERR_DEALS, index=False, sep=',', encoding='utf-8-sig')
print(f"Отчет об ошибках сделок: {OUTPUT_ERR_DEALS}")