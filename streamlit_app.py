import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.ttk as ttk
import math
from matplotlib.cm import get_cmap

# --- GLOBAL canvas holder ---
canvas = None

# Elements we want to plot
elements = ['T', 'TPM', 'Fmax', 'Fprum', 'H', 'SSV10M', 'D', 'P', 'SRA10M', 'SCEa', 'SCE']
BASE_URL = "https://opendata.chmi.cz/meteorology/climate/now/data/"

# --- Fetch station metadata from JSON ---
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
META_URL = f"https://opendata.chmi.cz/meteorology/climate/now/metadata/meta1-{yesterday}.json"

try:
    r = requests.get(META_URL)
    r.raise_for_status()
    meta_json = r.json()
    # Create dict: FULL_NAME -> WSI
    stations = {
        f"{row[2]} ({row[1]})": {
            "wsi": row[0],
            "elevation": row[5]
        }
        for row in meta_json['data']['data']['values']
    }
except Exception as e:
    messagebox.showerror("Error", f"Cannot fetch stations metadata: {e}")
    stations = {}


# --- REGIONS ---
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
            "Olešnice", "Obora", "Podivice (B1PODI01)", "Bukovinka",
            "Džbánice (B2DZBA01)", "Střelice", "Šatov (B2SATO01)"
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
            "Habry", "Krucemburk H3KRUC01", "Kadov", "Žďár nad Sázavou",
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
            "Vlkoš (B1VKLO01)", "Staré Hutě", "Hluk"
        ]
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

def find_station_wsi(partial_name):
    for full_name, wsi in stations.items():
        if partial_name.lower() in full_name.lower():
            return full_name, wsi
    return None, None

# --- Helper for centered axes ---
def centered_axis(ax, series, pad):
    if series is None or series.empty:
        return
    ax.set_ylim(series.min() - pad, series.max() + pad)

# --- Fetch and plot function ---
def fetch_and_plot(station_name):
    global canvas

    if station_name not in stations:
        messagebox.showerror("Error", f"Station {station_name} not found")
        return

    station_info = stations[station_name]
    wsi = station_info["wsi"]
    elevation = station_info["elevation"]

    # Last 3 days
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

    if combined_df.empty:
        messagebox.showerror("Error", "No data for this station")
        return

    df_pivot = combined_df.pivot(index='DT', columns='ELEMENT', values='VAL')

    # --- PLOT ---
    fig, ax_temp_left = plt.subplots(figsize=(16,6))
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
                linewidth=0.5,
                alpha=1.0,
                zorder=0
            )

    # --- X-axis setup ---
    combined_df.sort_values('DT', inplace=True)
    df_pivot = combined_df.pivot(index='DT', columns='ELEMENT', values='VAL')
    end_time = df_pivot.index.max()
    start_time = end_time - pd.Timedelta(hours=48)
    if start_time < df_pivot.index.min():
        start_time = df_pivot.index.min()
    if ax_temp: ax_temp.set_xlim(start_time, end_time)

    current_time = start_time.replace(minute=0, second=0, microsecond=0)
    current_time -= pd.Timedelta(hours=current_time.hour % 4)
    while current_time <= end_time:
        if ax_temp: ax_temp.axvline(x=current_time, color='lightblue', linestyle='--', linewidth=0.5, alpha=1.0, zorder=0)
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
        h = elevation  # altitude
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
        title = f"{station_name}, {elevation:.0f} m n. m."
    else:
        title = station_name

    plt.title(title)
    if ax_temp: ax_temp.set_xlabel("Time")
    fig.subplots_adjust(left=0.05, right=0.75, top=0.92, bottom=0.15)

    # --- Embed into Tkinter ---
    if canvas: canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def plot_region_element(region_key, element):
    region = regions[region_key]

    station_list = region["full"]
    if element == "SRA10M":
        station_list = region["full"] + region["precip_only"]

    global canvas
    fig, ax = plt.subplots(figsize=(16,6))

    # move y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    all_values = []
    all_times = []

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

        combined_df = pd.DataFrame()
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y%m%d") for i in [2,1,0]]

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

            df = df[df['ELEMENT'] == element]
            combined_df = pd.concat([combined_df, df])

        if combined_df.empty:
            continue

        df_pivot = combined_df.pivot(index='DT', columns='ELEMENT', values='VAL')
        if element not in df_pivot:
            continue

        # store valid data
        valid_series.append((df_pivot.index, df_pivot[element]))
        labels.append(station_partial)

        # collect values
        all_values.extend([y for y in df_pivot[element] if not pd.isna(y)])
        all_times.extend(df_pivot.index.tolist())

    if not valid_series:
        return

    # --- COLORS (NO WASTE) ---
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(valid_series)))

    # --- PLOTTING ---
    for i, ((x, y), label) in enumerate(zip(valid_series, labels)):
        ax.plot(x, y, label=label, color=colors[i])

    if not all_values or not all_times:
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
            ax.axhline(y=y, color='lightblue', linestyle='--', linewidth=0.5, zorder=0)

    # --- X-axis ---
    end_time = max(all_times)
    start_time = end_time - pd.Timedelta(hours=48)
    ax.set_xlim(start_time, end_time)

    current_time = start_time.replace(minute=0, second=0, microsecond=0)
    current_time -= pd.Timedelta(hours=current_time.hour % 4)

    while current_time <= end_time:
        ax.axvline(x=current_time, color='lightblue', linestyle='--', linewidth=0.5, zorder=0)
        current_time += pd.Timedelta(hours=4)

    def custom_time_formatter(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M\n%d.%m.%Y') if dt.hour == 0 else dt.strftime('%H:%M')

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,4,8,12,16,20]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_time_formatter))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # --- Labels ---
    nice_name = element_names.get(element, element)
    ax.set_title(f"{region_key} – {nice_name}")
    ax.set_xlabel("Time")

    ax.legend(fontsize=7, loc='upper left', ncol=3)

    # --- Tkinter ---
    if canvas:
        canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# --- GUI ---
root = tk.Tk()
root.title("ČHMÚ grafy stanic")
root.geometry("1550x850+200+100")

# Top frame for station dropdown
top_frame = tk.Frame(root)
top_frame.pack(side="top", fill="x", pady=10)

selected_station = tk.StringVar()

combobox = ttk.Combobox(
    top_frame,
    textvariable=selected_station,
    values=list(stations.keys()),
    width=60
)
combobox.pack(pady=5)
combobox.set("Brno, Žabovřesky (B2BZAB01)")

# --- Autocomplete filtering ---
def on_keyrelease(event):
    value = combobox.get().lower()
    if value == '':
        combobox['values'] = list(stations.keys())
    else:
        filtered = [s for s in stations.keys() if value in s.lower()]
        combobox['values'] = filtered

combobox.bind('<KeyRelease>', on_keyrelease)

btn_plot = tk.Button(top_frame, text="Zobraz", command=lambda: fetch_and_plot(selected_station.get()))
btn_plot.pack(pady=5)

# --- Region selection ---
region_frame = tk.Frame(root)
region_frame.pack(side="top", fill="x", pady=5)

region_inner = tk.Frame(region_frame)
region_inner.pack(anchor="center")

selected_region = tk.StringVar(value="JM")  # default region

for region_key in regions.keys():
    tk.Radiobutton(
        region_inner,
        text=region_key,
        variable=selected_region,
        value=region_key
    ).pack(side="left", padx=5)

# --- Element buttons ---
element_frame = tk.Frame(root)
element_frame.pack(side="top", fill="x", pady=5)

element_inner = tk.Frame(element_frame)
element_inner.pack(anchor="center")

elements_buttons = {
    "Teplota": "T",
    "T přízemní": "TPM",
    "Vítr avg": "Fprum",
    "Vítr nárazy": "Fmax",
    "Srážky": "SRA10M",
    "Vlhkost": "H"
}

for label, elem in elements_buttons.items():
    tk.Button(
        element_inner,
        text=label,
        width=8,
        command=lambda e=elem: plot_region_element(selected_region.get(), e)
    ).pack(side="left", padx=5, pady=5)

# --- Graph frame ---
graph_frame = tk.Frame(root)
graph_frame.pack(fill="both", expand=True)

root.mainloop()
