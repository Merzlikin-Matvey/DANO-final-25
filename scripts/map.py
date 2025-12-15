import folium
import pandas as pd
import requests
import time
import math
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
import datetime

infra_db = pd.read_csv('cache/cache_features.csv')
flats_clean = pd.read_csv('data/flats.csv')
deals_clean = pd.read_csv('data/deals.csv')

def remove_attribution(m):
    """
    Добавляет CSS-стиль к карте, который скрывает блок атрибуции.
    """
    macro = folium.Element("""
        <style>
            .leaflet-control-attribution {
                display: none !important;
            }
        </style>
    """)
    m.get_root().header.add_child(macro)



df = deals_clean

df['Дата ввода в эксплуатацию'] = pd.to_datetime(df['Дата ввода в эксплуатацию'])
df['Плановая дата РВЭ'] = pd.to_datetime(df['Плановая дата РВЭ'])
df['Дата договора (месяц.год)'] = pd.to_datetime(df['Дата договора (месяц.год)'])
df['Дата начала продаж'] = pd.to_datetime(df['Дата начала продаж'])

df['Дней после начала продаж'] = (df['Дата договора (месяц.год)'] - df['Дата начала продаж']).dt.days
df['Дней до ввода в эксплуатацию'] = (df['Дата ввода в эксплуатацию'] - df['Дата договора (месяц.год)']).dt.days
df['Дней между началом продаж и вводом в эксплуатацию'] = (df['Дата ввода в эксплуатацию'] - df['Дата начала продаж']).dt.days

df = df.sort_values(by='Дата договора (месяц.год)')
df['Задержка в днях'] = (
    df['Плановая дата РВЭ'] -
    df.groupby('ID корпуса')['Дата ввода в эксплуатацию'].transform('first')
).dt.days
df['Задержка'] = df['Задержка в днях'] > 0
df['Сдача в срок'] = df['Задержка в днях'] == 0
df['Ранняя сдача'] = df['Задержка в днях'] < 0

df['Площадь квартиры'] = df['Суммарная площадь сделок'] / df['Суммарное количество сделок'].replace(0, np.nan)

df['Уступка'] = df['Уступка'].apply(lambda val: val if isinstance(val, bool) else val == 'Договор уступки')

result = df.groupby('ID корпуса').agg(
    total=('Уступка', 'size'),
    assignments=('Уступка', 'sum')
)

assignment_rates = (
    df.groupby('Название ЖК')
    .agg(
        total=('Уступка', 'size'),
        assignments=('Уступка', 'sum')
    )
    .reset_index()
)

assignment_rates['Доля уступок'] = (assignment_rates['assignments'] / assignment_rates['total']) * 100

df = pd.merge(
    df,
    assignment_rates[['Название ЖК', 'Доля уступок']],
    on='Название ЖК',
    how='left'
)

def filter(df, condition):
    prev = len(df)
    df = df[condition]
    print(len(df) / prev)
    return df

print('Before:', len(df))
df = filter(df, df['Суммарное количество сделок'] == 1)
df = filter(df, df['Площадь квартиры'].notna() & (df['Площадь квартиры'] != 0))
df = filter(df, df['Цена квадратного метра'] < 500_000)
df = filter(df, ~np.isnan(df['Плановая дата РВЭ']))
df = filter(df, df['Плановая дата РВЭ'] <= datetime.datetime(2025, 11, 1))
df = filter(df, df['Расстояние до МКАД'] < 19)
df = filter(df, df['Общая проектная площадь'] <= 100_000)

print('After:', len(df))

MOSCOW_CENTER = (55.7558, 37.6173)

def get_new_moscow_boundaries():
    """Скачивает границы Новой Москвы (Nominatim)."""
    print("Скачивание границ Новой Москвы...")
    headers = {'User-Agent': 'PythonScript_RealEstate_Analysis/3.0'}
    base_url = "https://nominatim.openstreetmap.org/search"
    districts = ["Новомосковский административный округ", "Троицкий административный округ"]
    geo_json_list = []

    for district in districts:
        try:
            r = requests.get(base_url, params={'q': district, 'format': 'json', 'polygon_geojson': 1, 'limit': 1}, headers=headers)
            data = r.json()
            if data:
                geo_json_list.append({'name': district, 'geometry': data[0]['geojson']})
                print(f"-> {district}: OK")
        except Exception as e:
            print(f'Error: {e}')
            pass
        time.sleep(1.0)
    return geo_json_list

def calculate_offset_coords(points, offset_km):
    """
    Создает новые координаты, сдвинутые от центра на offset_km,
    сохраняя форму исходного полигона (МКАД).
    """
    center_lat, center_lon = MOSCOW_CENTER
    km_per_lat = 111.13
    km_per_lon = 111.32 * math.cos(math.radians(center_lat))

    new_ring = []
    for lat, lon in points:
        # Вектор от центра в километрах
        dy_km = (lat - center_lat) * km_per_lat
        dx_km = (lon - center_lon) * km_per_lon

        current_dist_km = math.sqrt(dx_km**2 + dy_km**2)
        if current_dist_km == 0: continue

        scale = (current_dist_km + offset_km) / current_dist_km

        new_dy_km = dy_km * scale
        new_dx_km = dx_km * scale

        new_lat = center_lat + (new_dy_km / km_per_lat)
        new_lon = center_lon + (new_dx_km / km_per_lon)

        new_ring.append([new_lat, new_lon])

    return new_ring

def process_mkad_geometry(infra_df):
    if infra_df is None or infra_df.empty or 'type' not in infra_df.columns: return None
    mkad_points = infra_df[infra_df['type'] == 'mkad'].copy()
    if mkad_points.empty: return None
    mkad_points['angle'] = mkad_points.apply(lambda r: math.atan2(r['lon'] - MOSCOW_CENTER[1], r['lat'] - MOSCOW_CENTER[0]), axis=1)
    return mkad_points.sort_values('angle')[['lon', 'lat']].values.tolist()

boundaries = get_new_moscow_boundaries()
mkad_ring_base = process_mkad_geometry(infra_db)

m = folium.Map(location=[55.55, 37.35], zoom_start=10)

# --- ОКРУГА ---
nm_layer = folium.FeatureGroup(name="Границы Адм. Округов", show=True)
if boundaries:
    for b in boundaries:
        folium.GeoJson(
            b['geometry'],
            name=b['name'],
            style_function=lambda x: {'color': '#444', 'weight': 1, 'fillOpacity': 0.05}
        ).add_to(nm_layer)
nm_layer.add_to(m)

# --- ЗОНЫ ---
zones_layer = folium.FeatureGroup(name="Зоны (0-6-12-19 км)", show=True)

if mkad_ring_base:
    ring_6 = calculate_offset_coords(mkad_ring_base, 6.0)
    ring_12 = calculate_offset_coords(mkad_ring_base, 12.0)
    ring_19 = calculate_offset_coords(mkad_ring_base, 19.0)

    # 2. МКАД
    folium.PolyLine(
        mkad_ring_base + [mkad_ring_base[0]],
        color='red', weight=4, opacity=0.8, tooltip="МКАД"
    ).add_to(zones_layer)

    # 3. 0-6 км
    folium.Polygon(
        locations=[ring_6, mkad_ring_base],
        color='#228B22', weight=2, dash_array='5, 5',
        fill=True, fill_color='#32CD32', fill_opacity=0.2,
        tooltip="Зона: 0 — 6 км"
    ).add_to(zones_layer)

    # 4. 6-12 км
    folium.Polygon(
        locations=[ring_12, ring_6],
        color='#00008B', weight=2, dash_array='5, 5',
        fill=True, fill_color='#1E90FF', fill_opacity=0.2,
        tooltip="Зона: 6 — 12 км"
    ).add_to(zones_layer)

    # 5. 12-19 км
    folium.Polygon(
        locations=[ring_19, ring_12],
        color='#800080',
        weight=2, dash_array='5, 5',
        fill=True, fill_color='#9370DB',
        fill_opacity=0.2,
        tooltip="Зона: 12 — 19 км"
    ).add_to(zones_layer)

    # 6. 19+ км
    folium.PolyLine(
        ring_19 + [ring_19[0]],
        color='#800080', weight=1, opacity=0.5, dash_array='2, 5',
        tooltip="Граница 19 км"
    ).add_to(zones_layer)

zones_layer.add_to(m)

# --- Flats ---
flats_layer = folium.FeatureGroup(name="ЖК (Flats)", show=False)
if not flats_clean.empty:
    for idx, row in flats_clean.iterrows():
        if pd.isna(row['lat']) or pd.isna(row['lon']): continue
        popup = f"<b>Flat</b><br>{row.get('clean_address','')}<br>{row['lat']:.4f}, {row['lon']:.4f}"
        folium.CircleMarker(
            [row['lat'], row['lon']], radius=4, color='#333', weight=1,
            fill=True, fill_color='#fff', fill_opacity=1.0, popup=folium.Popup(popup, max_width=300)
        ).add_to(flats_layer)
flats_layer.add_to(m)

# --- Deals ---
deals_layer = folium.FeatureGroup(name="Сделки (Deals)", show=False)
if not deals_clean.empty:
    for idx, row in deals_clean.iterrows():
        if pd.isna(row['Широта']) or pd.isna(row['Долгота']): continue
        popup = f"<b>Deal</b><br>{row.get('clean_address','')}<br>{row['Широта']:.4f}, {row['Долгота']:.4f}"
        folium.CircleMarker(
            [row['Широта'], row['Долгота']], radius=3, color='#333', weight=1,
            fill=True, fill_color='#00FFFF', fill_opacity=1.0, popup=folium.Popup(popup, max_width=300)
        ).add_to(deals_layer)
deals_layer.add_to(m)

# --- Инфраструктура ---
color_map = {
    'metro': "#7E1C50", 'train': "#2B0017", 'mkad': '#FF0000',
    'school': '#FFA500', 'park': '#32CD32', 'mall': '#800080'
}
if not infra_db.empty:
    unique_types = infra_db['type'].unique()
    infra_layers = {}

    for t in unique_types:
        if t == 'mkad': continue
        infra_layers[t] = folium.FeatureGroup(name=t, show=False)

    for idx, row in infra_db.iterrows():
        t = row['type']
        if t == 'mkad': continue
        color = color_map.get(t, 'gray')
        folium.CircleMarker(
            [row['lat'], row['lon']], radius=2, color=color,
            fill=True, fill_color=color, fill_opacity=0.7,
            popup=f"{t}<br>{row['name']}"
        ).add_to(infra_layers[t])

    for t, l in infra_layers.items():
        l.add_to(m)

remove_attribution(m)

folium.LayerControl(collapsed=False).add_to(m)

map_filename = "maps/map_infrastructure.html"
m.save(map_filename)
print(f"\nГотово! Карта сохранена: {map_filename}")


m = folium.Map(location=[55.55, 37.35], zoom_start=10)

nm_layer = folium.FeatureGroup(name="Границы Новой Москвы", show=True)

if 'boundaries' in locals() and boundaries:
    for b in boundaries:
        folium.GeoJson(
            b['geometry'],
            name=b['name'],
            style_function=lambda x: {
                'color': 'blue',
                'weight': 2,
                'fillColor': 'blue',
                'fillOpacity': 0.05
            },
            tooltip=b['name']
        ).add_to(nm_layer)
nm_layer.add_to(m)

# --- Flats ---
flats_layer = folium.FeatureGroup(name="ЖК (Flats)", show=False)

for idx, row in flats_clean.iterrows():
    if pd.isna(row['lat']) or pd.isna(row['lon']):
        continue

    addr = row['clean_address']
    popup_text = f"<b>ЖК / Flat</b><br>{addr}<br>{row['lat']}, {row['lon']}"

    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=7,
        color='black',
        weight=3,
        fill=True,
        fill_color="#FFFFFF",
        fill_opacity=1.0,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(flats_layer)

flats_layer.add_to(m)

remove_attribution(m)

folium.LayerControl(collapsed=False).add_to(m)

map_filename = "maps/flats_visualization.html"
m.save(map_filename)

print(f"\nКарта сохранена: {map_filename}")


df_grouped = df.groupby('Чистый адрес').agg({
    'Название ЖК': 'first', 'Широта': 'first', 'Долгота': 'first', 'Задержка в днях': 'mean', 'Расстояние до МКАД': 'mean'
}).reset_index().dropna(subset=['Широта', 'Долгота'])

# Коэффициенты в порядке убывания степени: [a, b, c] для ax^2 + bx + c
z = [-1.96, 30.04, -49.79]
p = np.poly1d(z)
print(f"Тренд (ручной ввод): {z[0]}x² + {z[1]}x + {z[2]}")

norm = colors.Normalize(vmin=-65, vmax=65)
try:
    cmap = matplotlib.colormaps['RdYlGn_r']
except:
    print("ALTER")
    cmap = plt.cm.RdYlGn_r

def get_color_for_delay(delay_val):
    return colors.to_hex(cmap(norm(delay_val)))


m = folium.Map(location=[55.53, 37.45], zoom_start=10, tiles='CartoDB positron')

nm_layer = folium.FeatureGroup(name="Границы Новой Москвы")
for b in boundaries:
    folium.GeoJson(b['geometry'], style_function=lambda x: {'color': '#444', 'weight': 1, 'fillOpacity': 0.05}).add_to(nm_layer)
nm_layer.add_to(m)

gradient_layer = folium.FeatureGroup(name="Градиент тренда")
if mkad_ring_base:
    folium.PolyLine(mkad_ring_base + [mkad_ring_base[0]], color='black', weight=2, opacity=0.8).add_to(gradient_layer)

    rings_cache = {0: mkad_ring_base}
    current_dist, end_km, step_km = 0, 19, 0.5

    while current_dist < end_km:
        next_dist = current_dist + step_km
        if next_dist not in rings_cache: rings_cache[next_dist] = calculate_offset_coords(mkad_ring_base, next_dist)

        predicted_delay = p((current_dist + next_dist) / 2)
        ring_color = get_color_for_delay(predicted_delay)

        folium.Polygon(
            locations=[rings_cache[next_dist], rings_cache[current_dist]],
            color=ring_color, weight=0, fill=True, fill_color=ring_color,
            fill_opacity=0.45,
            popup=f"Км: {current_dist}-{next_dist}<br>Тренд: {int(predicted_delay)} дн."
        ).add_to(gradient_layer)
        current_dist += step_km

    limit = rings_cache[max(rings_cache.keys())]
    folium.PolyLine(limit + [limit[0]], color='white', weight=1, dash_array='5,5').add_to(gradient_layer)
gradient_layer.add_to(m)

rokada_layer = folium.FeatureGroup(name="Рокадная линия метро")
rokada_coords = [
    (55.487706, 37.550621), (55.491907, 37.479969), (55.491182, 37.404025),
    (55.515712, 37.368424), (55.538943, 37.356576), (55.548839, 37.279474), (55.576123, 37.238102)
]
folium.PolyLine(rokada_coords, color="#000000", weight=3, opacity=0.8, dash_array='10, 10').add_to(rokada_layer)
rokada_layer.add_to(m)

markers_layer = folium.FeatureGroup(name="Точки ЖК")
for idx, row in df_grouped.iterrows():
    d = row['Задержка в днях']
    color = 'gray' if pd.isna(d) else ('#FF0000' if d > 0 else ('#00FF00' if d < 0 else '#0000FF'))

    popup = f"<b>{row['Название ЖК']}</b><br>Задержка: {int(d) if not pd.isna(d) else 'Нет'}"

    folium.CircleMarker(
        [row['Широта'], row['Долгота']], radius=6,
        color='white', weight=1.5,
        fill=True, fill_color=color, fill_opacity=1.0,
        popup=folium.Popup(popup, max_width=200)
    ).add_to(markers_layer)
markers_layer.add_to(m)

remove_attribution(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save("maps/gradient_map.html")
print("Контрастная темная карта сохранена: maps/gradient_map.html")

