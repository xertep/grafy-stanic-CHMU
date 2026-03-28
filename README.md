Tato aplikace využívá otevřená data Český hydrometeorologický ústav k zobrazení meteorologických grafů v tradičním, dobře známém stylu ze všech dostupných meteorologických stanic po celé republice.

Další funkce umožňují zobrazit různé meteorologické prvky z vybraných regionů (Jihomoravský, Zlínský, Vysočina) v jednom grafu.

Dále slouží k zobrazení všech dostupných textových předpovědí počasí v jednoduchém, tradičním stylu. Obsahuje předpovědi pro jednotlivé kraje, celou Českou republiku i vybraná pohoří.
Všechny předpovědi jsou psány lidskými meteorology a nejsou automaticky generovány z numerických modelů.


Hlavní soubor pro spuštění aplikace je streamlit_app.py. 
Funkční webová aplikace (vytvořená pomocí Streamlit) je dostupná zde: https://grafy-stanic-chmu-fknzcsm7chmxpvtwczmdyz.streamlit.app/

Aplikaci může při otevření trvat několik sekund, než se načte, zejména pokud byla delší dobu neaktivní je potřeba ji probudit kliknutím na modré tlačítko.
To je běžné chování — aplikace se při nečinnosti automaticky uspí a při dalším otevření potřebuje krátký čas na opětovné spuštění.


This app uses open data from the Czech Hydrometeorological Institute to display meteorological graphs in a familiar, classic style across all available weather stations in the country.

Additional features allow users to visualize various meteorological elements from selected regions (Jihomoravský, Zlínský, Vysočina) within a single graph.

Another section displays all available text weather forecasts in a simple, classic style. It includes forecasts for individual regions, the entire Czech Republic, and selected mountain ranges.
All forecasts are written by human meteorologists and are not automatically generated from weather models.

The main file to run the app is streamlit_app.py. 
A live version of the web application (built with Streamlit) is available here: https://grafy-stanic-chmu-fknzcsm7chmxpvtwczmdyz.streamlit.app/

The app may take a few seconds to load when you open it, especially if it has been inactive for a while (click of the button brings it back up).
This is normal behavior — the app automatically goes to sleep when not in use and needs a short moment to start again.
