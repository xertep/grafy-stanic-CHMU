import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import numpy as np
import math
from matplotlib.cm import get_cmap
from streamlit_extras.stylable_container import stylable_container
import re
from email.utils import parsedate_to_datetime
import pytz

from io import BytesIO
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw
import time
import gc

# ---------------- STATE INIT (TOP OF APP) ----------------
if "selected_element" not in st.session_state:
    st.session_state.selected_element = None

if "region_run" not in st.session_state:
    st.session_state.region_run = False


# Change browser tab title and favicon
st.set_page_config(
    page_title="Grafy stanic a předpovědi ČHMÚ",  # this changes the browser tab title
    page_icon="🌤️",                     # optional: emoji or path to an image
    layout="wide"                        # optional: wide layout for cards
)

# Your app content
# st.title("Grafy stanic ČHMÚ")

st.markdown("""
<style>
/* Reduce header but keep functionality */
header {
    visibility: visible;
}

/* Remove extra spacing */
.block-container {
    padding-top: 2.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
elements = ['T', 'TPM', 'Fmax', 'Fprum', 'H', 'SSV10M', 'D', 'P', 'SRA10M', 'SCEa', 'SCE']
BASE_URL = "https://opendata.chmi.cz/meteorology/climate/now/data/"


# --- REGIONS ---
regions = {
    "JM": {
        "full": [
            "Tišnov, Hájek", "Protivanov", "Ivanovice na Hané", "Brno, Žabovřesky",
            "Troubsko", "Brno, Tuřany", "Nemochovice", "Ždánice", "Pohořelice",
            "Kobylí", "Kuchařovice", "Brod nad Dyjí", "Strážnice", "Dyjákovice",
            "Lednice"
        ],
        "precip_only": [
            "Olešnice", "Obora", "Podivice", "Bukovinka",
            "Džbánice", "Střelice", "Šatov"
        ]
    },
    "VY": {
        "full": [
            "Svratouch", "Nedvězí", "Havlíčkův Brod", "Libice nad Doubravou",
            "Přibyslav, Hřiště", "Bystřice nad Pernštejnem", "Herálec", "Vatín",
            "Košetice", "Nový Rychnov", "Hubenov", "Jihlava, Hruškové Dvory",
            "Velké Meziříčí", "Černovice", "Počátky", "Sedlec", "Dukovany",
            "Moravské Budějovice"
        ],
        "precip_only": [
            "Habry", "Krucemburk", "Kadov", "Žďár nad Sázavou",
            "Nové Město na Moravě", "Humpolec", "Štoky", "Radostín",
            "Pacov", "Vysoké Studnice", "Kamenice nad Lipou, Vodná", "Třešť",
            "Brtnice", "Nová Ves", "Jemnice", "Náměšť nad Oslavou"
        ]
    },
    "ZL": {
        "full": [
            "Rožnov pod Radhoštěm", "Valašské Meziříčí", "Horní Bečva",
            "Velké Karlovice", "Bystřice pod Hostýnem", "Kateřinice, Ojičná",
            "Vsetín", "Hošťálková, Maruška", "Hošťálková", "Holešov",
            "Kroměříž", "Valašská Senice", "Zlín", "Vizovice",
            "Luhačovice, Kladná-Žilín", "Bojkovice", "Štítná nad Vláří",
            "Staré Město", "Strání", "Žítková", "Kašava, pod Rablinů",
            "Držková", "Nový Hrozenkov, Kohútka", "Velké Karlovice, Benešky",
            "Horní Bečva, Kudlačena"
        ],
        "precip_only": [
            "Valašská Bystřice", "Huslenky, Kychová", "Horní Lhota",
            "Vlkoš", "Staré Hutě", "Hluk"
        ]
    },
    "KV": {
        "full": [
            "Dyleň", "Krásné Údolí", "Mariánské Lázně, vodárna",
            "Aš", "Klínovec", "Karlovy Vary", "Karlovy Vary, Olšová Vrata",
            "Nejdek", "Šindelová, Obora", "Sokolov"
        ],
        "precip_only": [
            "Žlutice", "Abertamy", "Bečov nad Teplou",
            "Kynžvart, Lazy", "Luby", "Stráž nad Ohří"
        ]
    },
    "PL": {
        "full": [
            "Horská Kvilda", "Vlkonice", "Domažlice",
            "Hojsova Stráž", "Klatovy", "Nepomuk",
            "Plzeň, Bolevec", "Rokycany", "Špičák", "Staňkov",
            "Konstantinovy Lázně", "Kralovice",
            "Stříbro", "Tachov", "Zbiroh, Švabín"
        ],
        "precip_only": [
            "Filipova Huť", "Chanovice", "Nalžovské Hory",
            "Prášily", "Radošice", "Strašín", "Zámyšl",
            "Borovno, Míšov", "Čachrov", "Česká Kubice",
            "Horšovský Týn", "Chudenice", "Lovčice, Kvasetice",
            "Nezvěstice", "Pivoň", "Stod", "Železná Ruda",
            "Železná", "Bezvěrov", "Staré Sedlo, Darmyšl",
            "Liblín", "Planá", "Strašice",
            "Terešov", "Úlice"
        ]
    },
    "UL": {
        "full": [
            "Měděnec", "Milešovka", "Nová Ves v Horách",
            "Smolnice", "Sněžník", "Strojetice",
            "Teplice", "Ústí nad Labem, Vaňov",
            "Žatec", "Děčín", "Varnsdorf"
        ],
        "precip_only": [
            "Dubí", "Hřivice", "Hrob, Křižanov", "Klíny",
            "Libochovice, Poplze", "Mašťov",
            "Měrunice, Žichov", "Milešov",
            "Petrovice, Krásný Les", "Straškov-Vodochody",
            "Tisá", "Velká Černoc",
            "Děčín, Těchlovice", "Lobendava",
            "Šluknov", "Verneřice"
        ]
    },
    "SC": {
        "full": [
            "Rožmitál pod Třemšínem", "Čáslav, Nové město",
            "Chotusice, letiště", "Poděbrady", "Radovesnice II.",
            "Zbýšov, Dobrovítov", "Heřmanov", "Dobřichovice",
            "Kralupy nad Vltavou", "Lány", "Neumětely",
            "Příbram", "Brandýs nad Labem-St.B",
            "Dolní Bousov", "Katusice", "Semčice", "Tuhaň",
            "Nedrahovice, Rudolec", "Ondřejov", "Vavřinec, Žíšov"
        ],
        "precip_only": [
            "Březnice", "Bahno", "Dymokury", "Konárovice",
            "Kounov", "Karlova Ves", "Rakovník", "Hudlice",
            "Hvozdec, Mrtník", "Kamýk nad Vltavou",
            "Slaný", "Voznice", "Zlonice",
            "Boseň, Mužský", "Kostomlaty nad Labem",
            "Mšeno", "Káraný", "Benešov",
            "Řendějov, Nový Samechov", "Svatý Jan",
            "Horoměřice (Suchdol)", "Veleň, PČOV Miškovice"
        ]
    },
    "PH": {
        "full": [
            "Praha, Kbely", "Praha, Klementinum",
            "Praha, Ruzyně", "Praha, Vinohrady - Flora",
            "Praha, Komořany"
        ],
        "precip_only": [
            "Praha, Břevnov", "Praha, Běchovice",
            "Praha, Břevnov (Vypich)", "Praha, Bubeneč (F0 ÚČOV)",
            "Praha, Ďáblice (Ládví)", "Praha, Dubeč (PČOV Uhříněves)",
            "Praha, Hlubočepy (Barrandov)",
            "Praha, Horní Měcholupy (Kozinec)",
            "Praha, Horní Počernice", "Praha, Chodov",
            "Praha, Jinonice (Vidoule)", "Praha, Karlov",
            "Praha, Kyje", "Praha, Libeň (Prosek)",
            "Praha, Michle (Zelená Liška)",
            "Praha, Modřany sever II", "Praha, Radotín",
            "Praha, Řepy", "Praha, Stodůlky (Kopanina)",
            "Praha, Střešovice (Bruska)",
            "Praha, Vinohrady (Flora)",
            "Praha, Žižkov (Hrdlořezy)"
        ]
    },
    "CB": {
        "full": [
            "Borová Lada", "Černá v Pošumaví", "Husinec", "Ktiš, Tisovka",
            "Kvilda", "Sedlice", "Strakonice, Nové Strakonice", "Temelín, MW332",
            "Vimperk", "Volary", "Vráž", "Borkovice", "Byňov",
            "Český Krumlov, Přísečná", "Hlasivo",
            "Jindřichův Hradec, Děbolín", "Křemže, Mříč",
            "Nadějkov, Větrov", "Tábor, Měšice", "Třeboň, Lužnice",
            "Vyšší Brod", "Bučina, u Kvildy"
        ],
        "precip_only": [
            "Český Rudolec", "Bernartice", "Hluboká nad Vltavou",
            "Chelčice", "Jelení, Nová Pec", "Kestřany", "Krsice",
            "Orlík nad Vltavou", "Paseky", "Strážný", "Vacov, Peckov",
            "Volyně, Nihošovice", "Zbytiny", "Bechyně",
            "Dolní Dvořiště", "Chlum u Třeboně", "Nová Bystřice",
            "Netřebice", "Olší", "Pohorská Ves", "Staré Hutě",
            "Strmilov", "Stráž nad Nežárkou", "Trhové Sviny",
            "Mladá Vožice"
        ]
    },
    "LB": {
        "full": [
            "Desná, Souš", "Harrachov", "Holenice", "Bedřichov",
            "Česká Lípa", "Doksy", "Hejnice",
            "Jablonné v Podještědí", "Stráž pod Ralskem"
        ],
        "precip_only": [
            "Josefův Důl", "Lomnice nad Popelkou", "Roprachtice",
            "Semily", "Studenec", "Vysoké nad Jizerou",
            "Bedřichov, Hřebínek", "Bedřichov, Kristiánov",
            "Bedřichov, U Podkovy", "Hejnice, Kasárenská",
            "Kořenov, Jezdecká", "Kořenov, Jizerská cesta",
            "Kořenov, Kůrovec", "Kořenov, Lasičí",
            "Bedřichov, Černá hora", "Chotyně", "Chrastava",
            "Jablonec nad Nisou", "Křižany", "Mařenice",
            "Mimoň", "Nový Bor", "Nové Město pod Smrkem",
            "Višňová", "Zahrádky", "Žandov, Horní Police",
            "Bedřichov, Prameny Černé Nisy",
            "Bílý Potok, Pavlova cesta", "Bílý Potok, Smědava",
            "Bílý Potok, U Jeřábu", "Bedřichov, Tomšovka",
            "Bedřichov, Uhlířská", "Hejnice, Knajpa",
            "Hejnice, Smědavská hora"
        ]
    },
    "HK": {
        "full": [
            "Česká Skalice, Rozkoš", "Labská bouda", "Luční bouda",
            "Teplice nad Metují Zdoňov", "Trutnov", "Úpice",
            "Velichovky", "Vrchlabí", "Žacléř", "Borohrádek",
            "Deštné v Orlic. horách", "Orlické Záhoří - Vodárna",
            "Polom, Sedloňov", "Rokytnice v Orlic.horách",
            "Rychnov nad Kněžnou", "Broumov", "Holovousy",
            "Hradec Králové, Nový Hradec Králové", "Jičín",
            "Lázně Bělohrad", "Nový Bydžov",
            "Hradec Králové, Svobodné Dvory",
            "Adršpach, Horní Adršpach"
        ],
        "precip_only": [
            "Černý Důl", "Dolní Dvůr, Rudolfov", "Horní Maršov",
            "Hostinné", "Police nad Metují",
            "Pomezní boudy, Horní Malá Úpa", "Strážné",
            "Bílý Újezd, Hroška", "České Meziříčí",
            "Luisino údolí, Deštné v Orlických horách",
            "Olešnice, Čihálka", "Olešnice, Vodárna",
            "Slatina nad Zdobnicí", "Zdobnice", "Božanov",
            "Libáň", "Slatiny, Milíčeves"
        ]
    },
    "PU": {
        "full": [
            "Nedvězí", "Králíky", "Žamberk", "Gajer, Janov",
            "Hrušová", "Mokošín", "Pardubice, letiště",
            "Seč", "Skuteč", "Jevíčko", "Třebařov"
        ],
        "precip_only": [
            "Hradec nad Svitavou", "Polička", "Choceň",
            "Orličky", "Heřmanův Městec",
            "Hrochův Týnec", "Lubná", "Nové Hrady",
            "Červená Voda"
        ]
    },
    "OL": {
        "full": [
            "Prostějov", "Protivanov", "Bělotín", "Javorník",
            "Jeseník", "Dubicko", "Hanušovice",
            "Olomouc, Holice",
            "Staré Město pod Sněžníkem, Paprsek",
            "Šternberk", "Šumperk", "Přerov"
        ],
        "precip_only": [
            "Dřevohostice", "Kojetín",
            "Bělá pod Pradědem, Červenohorské sedlo",
            "Branná, Františkov",
            "Dlouhé Stráně, Kouty nad Desnou",
            "Hoštejn", "Velké Losiny", "Oskava"
        ]
    },
    "MS": {
        "full": [
            "Bílá, Konečná", "Bohumín, Záblatí",
            "Frýdek-Místek, Místek", "Frenštát pod Radhoštěm",
            "Slezská Harta", "Horní Lomná", "Jablunkov, Návsí",
            "Karviná", "Karlova Studánka", "Krnov",
            "Lučina", "Mořkov", "Mošnov",
            "Opava, Otice", "Ostrava, Poruba",
            "Ropice", "Světlá Hora", "Vítkov"
        ],
        "precip_only": ["Karlovice", "Lichnov", "Lomnice"]
    }
}


element_names = {
    "T": "Teplota (°C)",
    "TPM": "Teplota přízemní (°C)",
    "Fprum": "Vítr průměrný (m/s)",
    "Fmax": "Vítr nárazy (m/s)",
    "SRA10M": "Srážky 10 min (mm)",
    "H": "Vlhkost (%)",
    "SSV10M": "Sluneční svit (s)",
    "P": "Tlak (hPa)",
    "SCEa": "Sněhová pokrývka (cm)",
    "SCE": "Sněhová pokrývka (cm)"
}

# ---------------- LOAD STATIONS ----------------
@st.cache_data(ttl=0)
def load_stations():
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        url = f"https://opendata.chmi.cz/meteorology/climate/now/metadata/meta1-{yesterday}.json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        meta_json = r.json()

        return {
            f"{row[2]} ({row[1]})": {
                "wsi": row[0],
                "elevation": row[5]
            }
            for row in meta_json['data']['data']['values']
            if f"{row[2]} ({row[1]})" != "Reykjavik (ZIS04030)"
        }
    except Exception as e:
        st.error(f"Error loading stations: {e}")
        return {}

stations = load_stations()

# ---------------- DATA FETCH ----------------
def fetch_station_data(wsi):
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y%m%d") for i in [2,1,0]]
    combined_df = pd.DataFrame()

    for date_str in dates:
        url = f"{BASE_URL}10m-{wsi}-{date_str}.json"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                continue
            data = r.json()
        except:
            continue

        header = data['data']['data']['header'].split(',')
        values = data['data']['data']['values']

        df = pd.DataFrame(values, columns=header)
        df['DT'] = pd.to_datetime(df['DT'], utc=True)\
                       .dt.tz_convert('Europe/Prague')\
                       .dt.tz_localize(None)

        df['VAL'] = pd.to_numeric(df['VAL'], errors='coerce')
        df = df[df['ELEMENT'].isin(elements)]

        combined_df = pd.concat([combined_df, df])

    return combined_df


# ---------------- PLOT ----------------
def centered_axis(ax, series, pad):
    if series is None or series.empty:
        return
    ax.set_ylim(series.min() - pad, series.max() + pad)


def plot_station(df, station_name, elevation):
    if df.empty:
        st.error("No data available")
        return

    df_pivot = df.pivot(index='DT', columns='ELEMENT', values='VAL')

    # --- PLOT ---
    fig, ax_temp_left = plt.subplots(figsize=(16,6), dpi=150)
    ax_temp_left.set_yticks([])
    ax_temp_left.spines['left'].set_visible(False)

    # --- Temperature ---
    temp_series = None
    if 'T' in df_pivot and 'TPM' in df_pivot:
        temp_series = pd.concat([df_pivot['T'], df_pivot['TPM']])
    elif 'T' in df_pivot:
        temp_series = df_pivot['T']
    elif 'TPM' in df_pivot:
        temp_series = df_pivot['TPM']

    ax_temp = None
    if temp_series is not None and not temp_series.empty:
        ax_temp = ax_temp_left.twinx()
        ax_temp.spines['right'].set_position(('outward', 0))
        ax_temp.tick_params(axis='y', colors='red')
        
        if 'T' in df_pivot:
            ax_temp.plot(df_pivot.index, df_pivot['T'], color='red')
        if 'TPM' in df_pivot:
            ax_temp.plot(df_pivot.index, df_pivot['TPM'], color='#9636b6', linewidth=1)
        
        centered_axis(ax_temp, temp_series, 10)

        # --- Horizontal reference lines (every 5°C, based on visible axis) ---
        ymin, ymax = ax_temp.get_ylim()

        temp_min = math.floor(ymin / 5) * 5
        temp_max = math.ceil(ymax / 5) * 5

        for y in range(int(temp_min), int(temp_max) + 1, 5):
            ax_temp.axhline(
                y=y,
                color='lightblue',
                linestyle='--',
                linewidth=0.9,
                alpha=1.0,
                zorder=0
            )

    # --- X-axis setup ---
    df.sort_values('DT', inplace=True)
    df_pivot = df.pivot(index='DT', columns='ELEMENT', values='VAL')
    end_time = df_pivot.index.max()
    start_time = end_time - pd.Timedelta(hours=48)
    if start_time < df_pivot.index.min():
        start_time = df_pivot.index.min()
    if ax_temp: ax_temp.set_xlim(start_time, end_time)

    current_time = start_time.replace(minute=0, second=0, microsecond=0)
    current_time -= pd.Timedelta(hours=current_time.hour % 4)
    while current_time <= end_time:
        if ax_temp: ax_temp.axvline(x=current_time, color='lightblue', linestyle='--', linewidth=0.9, alpha=1.0, zorder=0)
        current_time += pd.Timedelta(hours=4)

    def custom_time_formatter(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M\n%d.%m.%Y') if dt.hour == 0 else dt.strftime('%H:%M')

    if ax_temp:
        ax_temp.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,4,8,12,16,20]))
        ax_temp.xaxis.set_major_formatter(plt.FuncFormatter(custom_time_formatter))
        plt.setp(ax_temp.get_xticklabels(), rotation=0, ha='center')

    # --- Wind ---
    wind_series = pd.concat([df_pivot['Fmax'], df_pivot['Fprum']]) if 'Fmax' in df_pivot and 'Fprum' in df_pivot else df_pivot['Fmax'] if 'Fmax' in df_pivot else df_pivot['Fprum'] if 'Fprum' in df_pivot else None
    ax_wind = None
    if wind_series is not None and not wind_series.empty:
        ax_wind = ax_temp_left.twinx()
        ax_wind.spines['right'].set_position(('outward', 30))
        if 'Fmax' in df_pivot: ax_wind.plot(df_pivot.index, df_pivot['Fmax'], color='#967b60')
        if 'Fprum' in df_pivot: ax_wind.plot(df_pivot.index, df_pivot['Fprum'], color='green')
        max_wind = df_pivot['Fmax'].max() if 'Fmax' in df_pivot else 15
        ax_wind.set_ylim(0, max(4, max_wind*1.2))
        ax_wind.tick_params(axis='y', colors='green')

    # --- Humidity ---
    ax_h = None
    if 'H' in df_pivot and not df_pivot['H'].dropna().empty:
        ax_h = ax_temp_left.twinx()
        ax_h.spines['right'].set_position(('outward', 60))
        ax_h.plot(df_pivot.index, df_pivot['H'], color='#09f8f8', linewidth=1)
        ax_h.set_ylim(0, 100)
        ax_h.tick_params(axis='y', colors='#09f8f8')

    # --- Sunshine ---
    ax_s = None
    if 'SSV10M' in df_pivot and not df_pivot['SSV10M'].dropna().empty:
        ax_s = ax_temp_left.twinx()
        ax_s.spines['right'].set_position(('outward', 90))
        ax_s.plot(df_pivot.index, df_pivot['SSV10M'], color='gold')
        ax_s.set_ylim(0, 601)
        ax_s.tick_params(axis='y', colors='gold')

    # --- Wind direction ---
    ax_d = None
    if 'D' in df_pivot and not df_pivot['D'].dropna().empty:
        ax_d = ax_temp_left.twinx()
        ax_d.spines['right'].set_position(('outward', 120))
        ax_d.scatter(df_pivot.index, df_pivot['D'], color='black', s=3)
        ax_d.set_ylim(0, 360)
        ax_d.tick_params(axis='y', colors='black')

    # --- Precipitation ---
    ax_r = None
    if 'SRA10M' in df_pivot and not df_pivot['SRA10M'].dropna().empty:
        ax_r = ax_temp_left.twinx()
        ax_r.spines['right'].set_position(('outward', 150))
        ax_r.plot(df_pivot.index, df_pivot['SRA10M'], color='blue')
        max_prec = df_pivot['SRA10M'].max()
        ax_r.set_ylim(0, max(4, max_prec*1.2))
        ax_r.tick_params(axis='y', colors='blue')
        daily_precip = df_pivot['SRA10M'].resample('D').sum()

        for day, total in daily_precip.items():
            if pd.isna(total) or total == 0:
                continue

            # Determine X position
            if day.date() == end_time.date():
                # Today → place at current time (end_time)
                x_pos = end_time
            else:
                # Previous days → place at end of that day
                x_pos = day + pd.Timedelta(hours=23, minutes=59)

            # Only draw if inside visible window
            if x_pos < start_time or x_pos > end_time:
                continue

            # Y position (top of precipitation axis)
            y_pos = ax_r.get_ylim()[1] * 0.99

            ax_r.text(
                x_pos,
                y_pos,
                f"{total:.1f}",
                color='blue',
                fontsize=9,
                ha='right',
                va='top'
            )

    # --- Snow (SCEa or SCE) ---
    ax_snow = None
    snow_column = None

    # Choose the snow column that exists
    if 'SCEa' in df_pivot and not df_pivot['SCEa'].dropna().empty:
        snow_column = 'SCEa'
    elif 'SCE' in df_pivot and not df_pivot['SCE'].dropna().empty:
        snow_column = 'SCE'

    if snow_column:
        ax_snow = ax_temp_left.twinx()
        ax_snow.spines['right'].set_position(('outward', 210))
        ax_snow.scatter(df_pivot.index, df_pivot[snow_column], color='#3eab8e', marker='D', s=60, zorder=5)
        ax_snow.set_ylim(0, max(5, df_pivot[snow_column].max() * 1.3))
        ax_snow.tick_params(axis='y', colors='#3eab8e')

    # --- Pressure ---
    pressure = None
    if 'P_hm' in df_pivot:
        pressure = df_pivot['P_hm']
    elif 'P' in df_pivot and 'T' in df_pivot:
        h = float(elevation) if elevation is not None else 0
        temp_K = df_pivot['T'] + 273.15
        pressure = df_pivot['P'] * ((1 - (0.0065 * h) / (temp_K + 0.0065 * h + 1)) ** -5.257)
    ax_p = None
    if pressure is not None and not pressure.empty:
        ax_p = ax_temp_left.twinx()
        ax_p.spines['right'].set_position(('outward', 180))
        ax_p.plot(df_pivot.index, pressure, color='black', linewidth=1)
        centered_axis(ax_p, pressure, 10)
        ax_p.tick_params(axis='y', colors='black')

    # --- Units (only if axis exists) ---
    if ax_temp: fig.text(0.75, 0.95, "°C", color='red')
    if ax_wind: fig.text(0.77, 0.95, "m/s", color='green')
    if ax_h: fig.text(0.80, 0.95, "%", color='#09f8f8')
    if ax_s: fig.text(0.83, 0.95, "s", color='gold')
    if ax_d: fig.text(0.855, 0.95, "°", color='black')
    if ax_r: fig.text(0.88, 0.95, "mm", color='blue')
    if ax_p: fig.text(0.905, 0.95, "hPa", color='black')
    if ax_snow: fig.text(0.93, 0.95, "cm", color='#3eab8e')

    # --- Finalize ---
    if elevation is not None:
        title = f"{station_name}, {float(elevation):.0f} m n. m."
    else:
        title = station_name

    plt.title(title)
    if ax_temp: ax_temp.set_xlabel("Time")
    fig.subplots_adjust(left=0.05, right=0.75, top=0.92, bottom=0.15)

    st.markdown('<div style="overflow-x: auto;">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

def find_station_wsi(partial_name):
    exact_match = None
    partial_match = None

    for full_name, info in stations.items():
        clean_name = full_name.split(" (")[0]

        if clean_name.lower() == partial_name.lower():
            exact_match = (full_name, info)
            break

        if partial_name.lower() in clean_name.lower():
            partial_match = (full_name, info)

    return exact_match or partial_match or (None, None)

def plot_region_element(region_key, element, regions, stations):
    region = regions[region_key]

    station_list = region["full"]
    if element == "SRA10M":
        station_list = region["full"] + region["precip_only"]

    fig, ax = plt.subplots(figsize=(16,6), dpi=150)

    # move y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    all_values = []
    all_times = []

    # ✅ NEW
    valid_series = []
    labels = []

    # --- DATA COLLECTION ---
    for station_partial in station_list:
        station_name, station_info = find_station_wsi(station_partial)
        if not station_info:
            continue

        wsi = station_info["wsi"]
        if not wsi:
            continue

        df = fetch_station_data(wsi)
        if df.empty:
            continue

        df = df[df['ELEMENT'] == element]
        if df.empty:
            continue

        df_pivot = df.pivot(index='DT', columns='ELEMENT', values='VAL')
        if element not in df_pivot:
            continue

        # ✅ STORE instead of plotting
        valid_series.append((df_pivot.index, df_pivot[element]))
        labels.append(station_partial)

        all_values.extend(df_pivot[element].dropna().tolist())
        all_times.extend(df_pivot.index.tolist())

    if not valid_series:
        st.warning("No data available for this selection")
        return

    # --- COLORS ---
    cmap = plt.get_cmap('tab20')
    base_colors = list(cmap.colors)

    extra_colors = [
        '#ffd700',  # gold
        '#ff1493',  # deep pink
        '#00ffff',  # cyan
        '#000000',  # black
        '#ff8c00',  # dark orange
    ]

    colors = base_colors + extra_colors

    # --- PLOTTING ---
    for i, ((x, y), label) in enumerate(zip(valid_series, labels)):
        ax.plot(
            x,
            y,
            label=label,
            color=colors[i % len(colors)]
        )

    if not all_values or not all_times:
        st.warning("No data available for this selection")
        return

    ymin = min(all_values)
    ymax = max(all_values)

    # --- Axis limits ---
    if element in ["SRA10M", "Fprum", "Fmax"]:
        ax.set_ylim(0, ymax * 1.1)
    else:
        if ymin != ymax:
            pad = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - pad, ymax + pad)

    ymin, ymax = ax.get_ylim()

    # --- Horizontal lines ---
    if element in ["T", "TPM"]:
        step = 5
    elif element in ["Fprum", "Fmax"]:
        step = 2
    elif element == "SRA10M":
        step = 1
    elif element == "H":
        step = 10
    else:
        step = None

    if step:
        y_start = math.floor(ymin / step) * step
        y_end = math.ceil(ymax / step) * step

        for y in np.arange(y_start, y_end + step, step):
            ax.axhline(y=y, color='lightblue', linestyle='--', linewidth=0.5)

    # --- X-axis ---
    end_time = max(all_times)
    start_time = end_time - pd.Timedelta(hours=48)
    ax.set_xlim(start_time, end_time)

    current_time = start_time.replace(minute=0, second=0, microsecond=0)
    current_time -= pd.Timedelta(hours=current_time.hour % 4)

    while current_time <= end_time:
        ax.axvline(x=current_time, color='lightblue', linestyle='--', linewidth=0.5)
        current_time += pd.Timedelta(hours=4)

    def custom_time_formatter(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M\n%d.%m.%Y') if dt.hour == 0 else dt.strftime('%H:%M')

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,4,8,12,16,20]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_time_formatter))

    # --- Labels ---
    nice_name = element_names.get(element, element)
    ax.set_title(f"{region_key} – {nice_name}")

    ax.legend(fontsize=7, loc='upper left', ncol=3)

    # --- Streamlit output ---
    st.markdown('<div style="overflow-x: auto;">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- TEXT FORECASTS FUNCTIONS ----------------

BASE_URL_forecasts = "https://opendata.chmi.cz/meteorology/weather/forecast/now/"

# Forecast types
REGION_FORECAST_TYPES = [
    ("pCK0tx", "Day 1"),
    ("pCKntx", "Night"),
    ("pCK1tx", "Day 2"),
    ("pCK2tx", "Day 3"),
    ("pCK3tx", "Day 4"),
    ("pCK4tx", "Day 5"),
]

CR_FORECAST_TYPES = [
    ("pCR8ts", "Meteorologická situace"),
    ("pCR0tx", "Dnes"),
    ("pCRntx", "Noc"),
    ("pCR1tx", "Zítra"),
    ("pCR2tx", "Pozítří"),
    ("pCR3tx", "3. den"),
    ("pCR4tx", "4. den"),
    ("pCR5tx", "5. den"),
    ("pCR8tx", "6.–8. den"),
]

MOUNTAIN_FORECAST_TYPES = [
    ("pCH1tx", "Day 1"),
    ("pCH2tx", "Day 2"),
]

mountains = [
    ("VY", "Žďárské vrchy"),
    ("ZL", "Javorníky a Bílé Karpaty"),
    ("CB", "Šumava a Novohradské hory"),
    ("HK", "Krkonoše"),
    ("LB", "Jizerské hory"),
    ("MT", "Beskydy"),
    ("OL", "Jeseníky a Králický Sněžník"),
    ("PL", "Český a Slavkovský les"),
    ("PU", "Orlické hory"),
    ("UL", "Krušné hory"),
]

CZECH_DAYS_GENITIVE = {
    0: "pondělí",
    1: "úterý",
    2: "středy",
    3: "čtvrtka",
    4: "pátku",
    5: "soboty",
    6: "neděle",
}


@st.cache_data(ttl=120)  # cache for 2 minutes
def get_forecast_listing():
    response = requests.get(BASE_URL_forecasts)
    return response.text

def transform_evening_to_night(file_time):
    days = ["pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle"]

    today = file_time
    tomorrow = today + timedelta(days=1)

    return f"Předpověď na noc {days[today.weekday()]}/{days[tomorrow.weekday()]}"

def format_update_time(utc_string):
    try:
        # parse UTC time
        dt_utc = datetime.fromisoformat(utc_string.replace("Z", "+00:00"))

        # convert to Czech local time
        local_tz = pytz.timezone("Europe/Prague")
        dt_local = dt_utc.astimezone(local_tz)

        # format: 16.4.2026 17:52
        return f"Aktualizováno {dt_local.day}.{dt_local.month}.{dt_local.year} {dt_local.strftime('%H:%M')} ⬆"
    except Exception:
        return None

def get_latest_file(pattern, html):
    matches = re.findall(
        r'(web_' + pattern + r'[^"]+\.json)</a>\s+(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})',
        html
    )

    if not matches:
        return None

    def parse_time(t):
        return datetime.strptime(t, "%d-%b-%Y %H:%M")

    matches.sort(key=lambda x: parse_time(x[1]))

    filename, time_str = matches[-1]

    dt = datetime.strptime(time_str, "%d-%b-%Y %H:%M")

    return BASE_URL_forecasts + filename, dt


def fetch_region(region_code):
    html = get_forecast_listing()
    place_name = None
    dalsi_dny_inserted = False
    all_data = []
    date_range_text = None

    forecast_types = CR_FORECAST_TYPES if region_code == "CR" else REGION_FORECAST_TYPES
    full_pattern_prefix = "" if region_code == "CR" else f"_RP{region_code}"

    for pattern, label in forecast_types:
        full_pattern = f"{pattern}{full_pattern_prefix}"
        result = get_latest_file(full_pattern, html)
        if not result:
            continue

        url, file_time = result

        try:
            data = requests.get(url).json()
            datum_vytvoreni = data.get("datumVytvoreni")
            features = data.get("data", {}).get("features", [])
            if not features:
                continue

            props = features[0].get("properties", {})

            if not place_name:
                place_name = props.get("place", {}).get(
                    "name",
                    "Česká republika" if region_code == "CR" else ""
                )

            headline_main = props.get("headline-main", {}).get("headline", "")
            items = sorted(props.get("data", []), key=lambda x: x.get("displayOrder", 0))

            all_data.append(
                (
                    pattern,
                    headline_main,
                    items,
                    props.get("senderName", ""),
                    file_time,
                    datum_vytvoreni
                )
            )

        except Exception as e:
            st.error(f"Error loading {label}: {e}")

    # region date range
    if region_code != "CR":
        pckn = next((x for x in all_data if x[0] == "pCKntx"), None)
        pck4 = next((x for x in all_data if x[0] == "pCK4tx"), None)

        if pckn and pck4:
            try:
                start_time = pckn[2][0].get("startTime")
                end_time = pck4[2][0].get("endTime")

                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

                start_day = CZECH_DAYS_GENITIVE[start_dt.weekday()]
                end_day = CZECH_DAYS_GENITIVE[end_dt.weekday()]

                date_range_text = (
                    f"od {start_day} {start_dt.day}.{start_dt.month}. "
                    f"do {end_day} {end_dt.day}.{end_dt.month}.{end_dt.year}"
                )

            except Exception:
                pass

    # regional duplicate day removal
    if region_code != "CR":
        pck1 = next((x for x in all_data if x[0] == "pCK1tx"), None)
        pck2 = next((x for x in all_data if x[0] == "pCK2tx"), None)

        if pck1 and pck2:
            h1 = " ".join(pck1[1].lower().split())
            h2 = " ".join(pck2[1].lower().split())

            if h1 == h2:
                all_data = [x for x in all_data if x[0] != "pCK2tx"]

    # CR remove outdated day 0
    if region_code == "CR":
        times = {p: t for p, _, _, _, t, _ in all_data}

        evening_update_detected = (
            "pCR0tx" in times and
            "pCRntx" in times and
            times["pCRntx"] > times["pCR0tx"]
        )

        if evening_update_detected:
            all_data = [x for x in all_data if x[0] != "pCR0tx"]

    # Region remove outdated day 0
    if region_code != "CR":
        times = {p: t for p, _, _, _, t, _ in all_data}

        evening_update_detected = (
            "pCK0tx" in times and
            "pCKntx" in times and
            times["pCKntx"] > times["pCK0tx"]
        )

        if evening_update_detected:
            all_data = [x for x in all_data if x[0] != "pCK0tx"]

    # CR duplicate headline removal (lower number wins)
    if region_code == "CR":
        seen = {}

        for entry in all_data:
            pattern, headline_main, items, sender, t, created = entry

            if not headline_main:
                seen[pattern] = entry
                continue

            key = " ".join(headline_main.lower().split())

            if key in seen:
                prev_pattern = seen[key][0]

                def get_index(p):
                    m = re.search(r"pCR(\d+)", p)
                    return int(m.group(1)) if m else 999

                if get_index(pattern) < get_index(prev_pattern):
                    seen[key] = entry
            else:
                seen[key] = entry

        all_data = list(seen.values())

        order = {p: i for i, (p, _) in enumerate(CR_FORECAST_TYPES)}
        all_data.sort(key=lambda x: order.get(x[0], 999))

    # build output
    output_lines = []

    if place_name:
        output_lines.append(f'<h4>=== Předpověď {place_name} ===</h4><br>')
        if date_range_text:
            output_lines.append(f'{date_range_text}<br>')

    for pattern, headline_main, items, sender, t, created in all_data:
        if pattern in ["pCK2tx", "pCK3tx", "pCK4tx"] and not dalsi_dny_inserted:
            output_lines.append('<br><h4>=== Další dny ===</h4><br>')
            dalsi_dny_inserted = True

        # show every main headline exactly as received
        if headline_main:
            output_lines.append(f'<br><h5>{headline_main}</h5><br>')

        for item in items:
            item_name = item.get("name")
            item_headline = item.get("headline")
            item_text = item.get("displayText")

            normalized_main = " ".join(headline_main.lower().split()) if headline_main else ""
            normalized_item = " ".join(item_headline.lower().split()) if item_headline else ""

            if item_name == "textIntro" and not item_headline:
                output_lines.append('<br><b>Stručné shrnutí počasí:</b><br>')

            elif item_name == "textWeather" and not item_headline:
                output_lines.append('<br><b>Počasí (00-24):</b><br>')

            elif item_headline and normalized_item != normalized_main:
                output_lines.append(f'<br><b>{item_headline}</b><br>')

            if item_text:
                output_lines.append(f'{item_text.replace("\\xa0", " ")}<br>')

        # region meteorologist
        if pattern == "pCK1tx" and sender:
            if created:
                formatted = format_update_time(created)
                if formatted:
                    output_lines.append(f'<br>{formatted}<br>')
            output_lines.append(f'Meteorolog: {sender}<br>')

        # CR meteorologists
        if region_code == "CR":
            if pattern == "pCR1tx" and sender:
                if created:
                    formatted = format_update_time(created)
                    if formatted:
                        output_lines.append(f'<br>{formatted}<br>')
                output_lines.append(f'Meteorolog: {sender}<br>')
                output_lines.append('<br><b>=== Další dny ===</b><br>')

            if pattern in ["pCR4tx", "pCR8tx", "pCR8ts"] and sender:
                if created:
                    formatted = format_update_time(created)
                    if formatted:
                        output_lines.append(f'<br>{formatted}<br>')
                output_lines.append(f'Meteorolog: {sender}<br>')

    # final regional sender
    for pattern, _, _, sender, _, created in reversed(all_data):
        if pattern == "pCK4tx" and sender:
            if created:
                formatted = format_update_time(created)
                if formatted:
                    output_lines.append(f'<br>{formatted}<br>')
            output_lines.append(f'Meteorolog: {sender}<br>')
            break

    return "".join(output_lines)


def fetch_mountain(mountain_code):
    html = get_forecast_listing()
    sender_name = None
    place_name = None
    output_lines = []
    update_time = None

    for pattern, label in MOUNTAIN_FORECAST_TYPES:
        full_pattern = f"{pattern}_RP{mountain_code}"
        result = get_latest_file(full_pattern, html)
        if not result:
            continue

        url, _ = result
        
        try:
            data = requests.get(url).json()
            datum_vytvoreni = data.get("datumVytvoreni")
            features = data.get("data", {}).get("features", [])
            if not features:
                continue
            props = features[0].get("properties", {})
            if not place_name:
                place_name = props.get("place", {}).get("name", "")
            if not sender_name:
                sender_name = props.get("senderName", "")

            if pattern == "pCH2tx":
                update_time = datum_vytvoreni

            headline_main = props.get("headline-main", {}).get("headline", "")
            items = sorted(props.get("data", []), key=lambda x: x.get("displayOrder", 0))

            if headline_main:
                output_lines.append(f'<br><b>{headline_main}</b><br>')

            for item in items:
                h = item.get("headline")
                text = item.get("displayText")
                if h:
                    output_lines.append(f'<br><b>{h}</b><br>')
                if text:
                    text = text.replace("\xa0", " ")
                    output_lines.append(f'{text}<br>')

        except Exception as e:
            st.error(f"Error loading {label} ({mountain_code}): {e}")

    if place_name:
        output_lines.insert(0, f'<b>=== Předpověď {place_name} ===</b><br>')
    if sender_name:
        if update_time:
            formatted = format_update_time(update_time)
            if formatted:
                output_lines.append(f'<br>{formatted}<br>')
        output_lines.append(f'Meteorolog: {sender_name}<br>')

    return "".join(output_lines)



# ---------------- UI ----------------
st.title("ČHMÚ meteostanice a předpovědi počasí")

# ---------------- MODE ----------------
mode = st.radio("Zvol režim", ["Stanice", "Region", "Textové předpovědi", "Mapy Aladin", "Radar"])

if "last_mode" not in st.session_state:
    st.session_state.last_mode = None

if st.session_state.last_mode != mode:

    # leaving Region → always reset region state
    if st.session_state.last_mode == "Region":
        st.session_state.selected_element = None
        st.session_state.region_run = False

    st.session_state.last_mode = mode

# ---------------- STATION MODE ----------------
if mode == "Stanice":

    st.subheader("Graf vybrané stanice")

    station_list = list(stations.keys())

    default_station = "Brno, Žabovřesky (B2BZAB01)"
    default_index = station_list.index(default_station) if default_station in station_list else 0

    station_name = st.selectbox(
        "Vyber stanici",
        station_list,
        index=default_index
    )

    # --- Shortcut buttons ---
    col1, col2, col3 = st.columns([2, 1, 1])  # main button + 2 shortcuts

    with col1:
        btn_main = st.button("Zobraz data")

    with col2:
        btn_dukovany = st.button("Dukovany")

    with col3:
        btn_brno = st.button("Brno, Žabovřesky")

    # --- Decide which station to show ---
    if btn_dukovany:
        chosen_station = "Dukovany (B2DUKO01)"
    elif btn_brno:
        chosen_station = "Brno, Žabovřesky (B2BZAB01)"
    elif btn_main:
        chosen_station = station_name
    else:
        chosen_station = None

    # 👇 PLACEHOLDER HERE (after button!)
    station_placeholder = st.empty()

    # --- Plot section ---
    if chosen_station:
        station_info = stations[chosen_station]
        wsi = station_info["wsi"]
        elevation = station_info["elevation"]

        with st.spinner("Načítám data..."):
            df = fetch_station_data(wsi)

        with station_placeholder.container():
            plot_station(df, chosen_station, elevation)

    else:
        station_placeholder.markdown(
            "<p style='color:#777;'>Zobrazí graf vybrané stanice</p>",
            unsafe_allow_html=True
        )

# ---------------- REGION MODE ----------------
elif mode == "Region":

    st.subheader("Přehled počasí v krajích")

    selected_region = st.segmented_control(
        "Kraj",
        list(regions.keys()),
        default=list(regions.keys())[0]
    )

    elements_buttons = {
        "Teplota": "T",
        "T přízemní": "TPM",
        "Vítr avg": "Fprum",
        "Vítr nárazy": "Fmax",
        "Srážky": "SRA10M",
        "Vlhkost": "H"
    }

    selected_element_label = st.segmented_control(
        "Prvek",
        list(elements_buttons.keys()),
        key="region_element_selector"
    )

    selected_element = elements_buttons.get(selected_element_label)

    # 👇 INIT
    if "region_run" not in st.session_state:
        st.session_state.region_run = False

    if "last_selected_element" not in st.session_state:
        st.session_state.last_selected_element = None

    # 👇 Detect CHANGE (this is the key)
    if selected_element != st.session_state.last_selected_element:
        st.session_state.region_run = True
        st.session_state.last_selected_element = selected_element

    region_placeholder = st.empty()

    # ---------------- OUTPUT ----------------
    if st.session_state.region_run and selected_element:

        with st.spinner("Načítám data..."):
            plot_region_element(
                selected_region,
                selected_element,
                regions,
                stations
            )

        st.session_state.region_run = False

    else:
        region_placeholder.markdown(
            "<p style='color:#777;'>Zobrazí vybraný prvek pro všechny dostupné stanice v kraji do jednoho grafu</p>",
            unsafe_allow_html=True
        )

# ---------------- FORECAST MODE ----------------
elif mode == "Textové předpovědi":

    st.subheader("Předpovědi počasí ČHMÚ")

    # ---------------- STEP 1 ----------------
    mode_choice = st.segmented_control(
        "Co chceš zobrazit?",
        ["Kraje a ČR", "Horské oblasti"],
        key="forecast_type"
    )

    active = None

    # ---------------- REGIONS ----------------
    if mode_choice == "Kraje a ČR":

        st.markdown("### Kraje a ČR")

        region_codes = ["JM","VY","ZL","KV","PL","UL","SC","PH","CB","LB","HK","PU","OL","MS","CR"]
        region_codes_cz = ["JM","VY","ZL","KV","PL","UL","SC","PH","CB","LB","HK","PU","OL","MS","ČR"]

        region_map = dict(zip(region_codes_cz, region_codes))

        selected_region_label = st.segmented_control(
            "Vyber kraj",
            list(region_map.keys()),
            key="region_sel"
        )

        if selected_region_label:
            active = ("region", region_map[selected_region_label])

    # ---------------- MOUNTAINS ----------------
    elif mode_choice == "Horské oblasti":

        st.markdown("### Horské oblasti")

        mountain_map = {code: code for code, _ in mountains}

        selected_mountain = st.segmented_control(
            "Vyber oblast",
            list(mountain_map.keys()),
            key="mountain_sel"
        )

        if selected_mountain:
            active = ("mountain", selected_mountain)

    # 👇 MOVE PLACEHOLDER HERE (IMPORTANT)
    forecast_placeholder = st.empty()

    # ---------------- OUTPUT ----------------
    with st.spinner("Načítám data..."):

        if active is None:
            forecast_placeholder.markdown(
                "<p style='color:#777;'>Vyber konkrétní oblast</p>",
                unsafe_allow_html=True
            )
            st.stop()

        kind, value = active

        if kind == "mountain":
            forecast_html = fetch_mountain(value)
        else:
            forecast_html = fetch_region(value)

    forecast_placeholder.markdown(forecast_html, unsafe_allow_html=True)

# ---------------- PRECIP MODE ----------------
elif mode == "Mapy Aladin":

    st.markdown("### Víc map z Aladina najdete tady:")
    st.markdown("[https://aladin-open-data-chmu-t7goat3t8imstn56yc6ash.streamlit.app/](https://aladin-open-data-chmu-t7goat3t8imstn56yc6ash.streamlit.app/)")

    st.markdown("#####  ")
    st.markdown(
        '##### 24h srážky ▼',
        unsafe_allow_html=True
    )

    BASE_URL_FLOODS = "https://opendata.chmi.cz/meteorology/floods/"

    @st.cache_data(ttl=60, show_spinner=False)
    def get_last_runs_from_server(n=8):
        url = "https://opendata.chmi.cz/meteorology/floods/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            pattern = re.compile(r"floods_prec24h_(\d{10})\+(\d+)\.png")
            matches = pattern.findall(response.text)

            if not matches:
                return []

            newest_run, _ = max(
                matches,
                key=lambda x: (
                    datetime.strptime(x[0], "%Y%m%d%H"),
                    int(x[1])
                )
            )

            newest_dt = datetime.strptime(newest_run, "%Y%m%d%H")

            return [
                (newest_dt - timedelta(hours=6*i)).strftime("%Y%m%d%H")
                for i in range(n)
            ]

        except Exception:
            return []

    runs = get_last_runs_from_server(8)

    if not runs:
        st.error("Nepodařilo se načíst dostupné běhy modelu.")
        st.stop()

    # --- Selector ---
    selected_run = st.segmented_control(
        "Vyber běh modelu",
        runs,
        format_func=lambda x: f"{x[6:8]}.{x[4:6]}. {x[8:]} UTC",
        default=runs[0]
    )

    # --- Steps ---
    steps = [24,30,36,42,48,54,60,66,72]

    run_dt = datetime.strptime(selected_run, "%Y%m%d%H")

    czech_days = [
        "pondělí", "úterý", "středy", "čtvrtka",
        "pátku", "soboty", "neděle"
    ]

    for step in steps:
        img_url = f"{BASE_URL_FLOODS}floods_prec24h_{selected_run}+{step}.png"
        forecast_time = run_dt + timedelta(hours=step)

        valid_time = (
            pd.Timestamp(forecast_time, tz="UTC")
              .tz_convert("Europe/Prague")
              .strftime("%d.%m. %H:%M")
        )

        forecast_dt = pd.Timestamp(forecast_time, tz="UTC").tz_convert("Europe/Prague")

        day_name = czech_days[forecast_dt.weekday()]

        st.markdown(
            f"<div style='font-weight:500; margin-bottom:2px;'>"
            f"24h suma srážek do {day_name} {valid_time} hod ▼</div>",
            unsafe_allow_html=True
        )

        st.image(img_url, use_container_width=False)
        st.write("")

# ---------------- RADAR MODE ----------------
elif mode == "Radar":

    st.subheader("Radar")

    BASE_URL = "https://opendata.chmi.cz/meteorology/weather/radar/composite/maxz/png_masked/"
    FULL_EXTENT = [11.267, 20.770, 48.047, 52.167]
    DATA_EXTENT = [11.267, 19.624, 48.047, 51.458]

    MARKERS = [
        (16.1113, 49.0546),
        (16.5642, 49.2147)
    ]

    @st.cache_data(ttl=60, show_spinner=False)
    def get_latest_radar_files(n=20):
        try:
            response = requests.get(BASE_URL, timeout=10)
            response.raise_for_status()

            pattern = re.compile(
                r"pacz2gmaps3\.z_max3d\.(\d{8}\.\d{4})\.0\.png"
            )

            matches = pattern.findall(response.text)

            if not matches:
                return []

            latest = sorted(matches)[-n:]

            return [
                f"pacz2gmaps3.z_max3d.{ts}.0.png"
                for ts in latest
            ]

        except Exception:
            return []


    @st.cache_data(ttl=60, show_spinner="Načítám data...")
    def download_radar_bytes(file_urls):
        session = requests.Session()
        images = []

        for url in file_urls:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            images.append(response.content)

        return images


    @st.cache_resource
    def load_border_overlay():
        return Image.open("border_overlay.png").convert("RGBA")


    def lonlat_to_pixel(extent, lon, lat, width, height):
        x = int((lon - extent[0]) / (extent[1] - extent[0]) * width)
        y = int((extent[3] - lat) / (extent[3] - extent[2]) * height)
        return x, y


    def build_gif_from_bytes(image_bytes_list, border_overlay):
        frames = []

        first_img = Image.open(BytesIO(image_bytes_list[0])).convert("RGBA")
        width, height = first_img.size

        # position of actual radar area inside full image
        x1, y1 = lonlat_to_pixel(FULL_EXTENT, DATA_EXTENT[0], DATA_EXTENT[3], width, height)
        x2, y2 = lonlat_to_pixel(FULL_EXTENT, DATA_EXTENT[1], DATA_EXTENT[2], width, height)

        map_w = x2 - x1
        map_h = y2 - y1

        # resize border overlay only once
        overlay_small = border_overlay.resize((map_w, map_h))

        overlay_full = Image.new("RGBA", first_img.size, (0, 0, 0, 0))
        overlay_full.paste(overlay_small, (x1, y1), overlay_small)

        for img_bytes in image_bytes_list:
            radar_img = Image.open(BytesIO(img_bytes)).convert("RGBA")

            white_bg = Image.new("RGBA", radar_img.size, "white")
            white_bg.paste(radar_img, (0, 0), radar_img)

            combined = Image.alpha_composite(white_bg, overlay_full)

            draw = ImageDraw.Draw(combined)

            for lon, lat in MARKERS:
                x, y = lonlat_to_pixel(FULL_EXTENT, lon, lat, width, height)

                size = 4
                draw.line((x - size, y, x + size, y), fill="#02ebdb", width=2)
                draw.line((x, y - size, x, y + size), fill="#02ebdb", width=2)

            frames.append(combined)

        buf = BytesIO()

        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=400,
            loop=0
        )

        buf.seek(0)
        return buf.getvalue()


    def format_time(filename):
        ts = filename.split(".")[2] + filename.split(".")[3]
        dt = datetime.strptime(ts, "%Y%m%d%H%M")
        dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Prague"))

        return dt.strftime("%d.%m.%y %H:%M")


    border_overlay = load_border_overlay()

    if st.button("🔄 Aktualizovat radar"):
        st.cache_data.clear()
        st.rerun()

    radar_files = get_latest_radar_files()

    if not radar_files:
        st.error("Nepodařilo se načíst radarová data.")
        st.stop()

    file_urls = [BASE_URL + f for f in radar_files]

    image_bytes = download_radar_bytes(file_urls)

    gif_data = build_gif_from_bytes(image_bytes, border_overlay)
    gc.collect()

    st.caption(f"Aktuální radar: od {format_time(radar_files[0])} do {format_time(radar_files[-1])} hod ▼")

    st.image(gif_data, use_container_width=True)

