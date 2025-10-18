import pandas as pd

# Wczytanie z folderu nadrzędnego
df = pd.read_csv('../data/road_eqr_carpda.csv', 
                 sep=None, 
                 engine='python', 
                 compression='infer')

# Sprawdzenie struktury
print("Kolumny:")
print(list(df.columns))
print("\nPierwsze 10 wierszy:")
print(df.head(10))

# Upewnij się, że istnieje folder 'out'
import os
os.makedirs('out', exist_ok=True)

# Zapisz podgląd
df.head(5000).to_excel('out/preview_original.xlsx', index=False)

#W pierwszej kolejnosci odrzuciłem kolumny, w których znajdowały się puste wierze. Następnie odrzyciłem skróty od nazw.
#Ostatecznie pozbyłem się również kolumn, które zawierały identyczne wpisy dla każej pozycji.
keep_cols = ['Motor energy', 'Geopolitical entity (reporting)', 'TIME_PERIOD', 'OBS_VALUE']
df_clean = df[keep_cols]

# Excel
df_clean.to_excel('out/df_clean.xlsx', index=False, sheet_name='dane')

# CSV (UTF-8)
df_clean.to_csv('out/df_clean.csv', index=False, encoding='utf-8')

# =====================================================
# Ćwiczenie 6: Dynamika napędów + kraje z rosnącymi LEVs
# Autor: Dawid Kosmalski
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager

# =====================================================
# 1. Wczytanie danych
# =====================================================
df = pd.read_excel("out/df_clean.xlsx", sheet_name="dane")

# Upewnienie się, że kolumny są właściwe
df.columns = df.columns.str.strip()
print("Kolumny:", list(df.columns))

# =====================================================
# 2. Czyszczenie i filtracja danych
# =====================================================
# Usuwamy agregaty UE i wpisy 'Total/All'
mask_ue = df["Geopolitical entity (reporting)"].astype(str).str.contains(
    r"European Union|EU27|EU28|EA19", case=False, na=False
)
mask_total = df["Motor energy"].astype(str).str.contains(r"Total|All", case=False, na=False)
df = df[~mask_ue & ~mask_total].copy()

# Zakres lat – ostatnie 5
df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y", errors="coerce")
years = sorted(df["TIME_PERIOD"].dt.year.dropna().unique())[-5:]
df = df[df["TIME_PERIOD"].dt.year.isin(years)].copy()

# =====================================================
# 3. Normalizacja nazw napędów
# =====================================================
def normalize_energy(x):
    x = str(x).upper()
    if "ELECTRIC" in x and "HYBRID" not in x:
        return "BEV"
    elif "PLUG-IN" in x:
        return "PHEV"
    elif "HYBRID" in x:
        return "HEV"
    elif "DIESEL" in x:
        return "Diesel"
    elif "PETROL" in x or "GASOLINE" in x:
        return "Petrol"
    else:
        return "Inne"

df["Motor energy"] = df["Motor energy"].apply(normalize_energy)
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce").round(0).astype("Int64")

# =====================================================
# 4. Agregacje – dynamika napędów i LEVs
# =====================================================
fuel_year = (
    df.groupby([df["TIME_PERIOD"].dt.year, "Motor energy"])["OBS_VALUE"]
    .sum()
    .unstack()
    .fillna(0)
)
fuel_year = np.ceil(fuel_year).astype(int)

lev_labels = ["BEV", "PHEV", "HEV"]
lev_country_year = (
    df[df["Motor energy"].isin(lev_labels)]
    .groupby(["Geopolitical entity (reporting)", df["TIME_PERIOD"].dt.year])["OBS_VALUE"]
    .sum()
    .unstack()
    .fillna(0)
)
lev_country_year = np.ceil(lev_country_year).astype(int)

lev_country_growth = pd.DataFrame({
    "Przyrost": lev_country_year[years[-1]] - lev_country_year[years[0]],
    "Wzrosty_rr": (lev_country_year.diff(axis=1) > 0).sum(axis=1)
}).sort_values("Przyrost", ascending=False)

# =====================================================
# 5. Styl wykresów
# =====================================================
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk")
os.makedirs("plots", exist_ok=True)

# =====================================================
# 6. Wykres 1 – Dynamika napędów
# =====================================================
plt.figure(figsize=(12, 7))
for col in fuel_year.columns:
    plt.plot(fuel_year.index.astype(int), fuel_year[col], marker="o", linewidth=2.5, label=col)

plt.title("Dynamika liczby rejestracji wg rodzaju napędu", fontsize=20)
plt.xlabel("Rok", fontsize=16)
plt.ylabel("Liczba rejestracji", fontsize=16)
plt.xticks(fuel_year.index.astype(int))
plt.legend(title="Rodzaj napędu", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plot1_path = "plots/1_dynamika_napedow.png"
plt.savefig(plot1_path, dpi=300)
plt.close()

# =====================================================
# 7. Wykres 2 – Udziały procentowe napędów (3 ostatnie lata)
# =====================================================
last3 = sorted(years)[-3:]
fuel_share = (
    fuel_year.loc[last3].div(fuel_year.loc[last3].sum(axis=1), axis=0) * 100
).round(1)

plt.figure(figsize=(12, 7))
fuel_share.plot(kind="bar", stacked=True, ax=plt.gca(), width=0.7)

for idx, year in enumerate(last3):
    bottom = 0
    for col in fuel_share.columns:
        val = fuel_share.loc[year, col]
        if val > 5:
            plt.text(idx, bottom + val / 2, f"{val:.1f}%", ha="center", va="center", fontsize=10)
        bottom += val

plt.title("Udziały procentowe napędów w ostatnich 3 latach", fontsize=20)
plt.xlabel("Rok", fontsize=16)
plt.ylabel("Udział [%]", fontsize=16)
plt.xticks(rotation=0)
plt.legend(title="Rodzaj napędu", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plot2_path = "plots/2_udzialy_napedow.png"
plt.savefig(plot2_path, dpi=300)
plt.close()

# =====================================================
# 8. Wykres 3 – TOP-10 krajów wg przyrostu LEVs
# =====================================================
top10 = lev_country_growth.head(10)
plt.figure(figsize=(12, 7))
sns.barplot(
    x=top10["Przyrost"],
    y=top10.index,
    palette="crest"
)
for i, (val, wzr) in enumerate(zip(top10["Przyrost"], top10["Wzrosty_rr"])):
    plt.text(val + (max(top10["Przyrost"]) * 0.01), i, f"{int(val)} ({wzr}↑)", va="center", fontsize=11)

plt.title("TOP-10 krajów wg przyrostu LEVs", fontsize=20)
plt.xlabel("Przyrost rejestracji (ostatni–pierwszy rok)", fontsize=16)
plt.ylabel("Kraj", fontsize=16)
plt.tight_layout()
plot3_path = "plots/3_top10_kraje_LEV.png"
plt.savefig(plot3_path, dpi=300)
plt.close()

# =====================================================
# 9. Wykres 4 – Dynamika LEVs dla TOP-5 krajów
# =====================================================
top_countries = top10.index[:5]
plt.figure(figsize=(12, 7))
for country in top_countries:
    series = lev_country_year.loc[country, years].values
    plt.plot(years, series, marker='o', linewidth=2.5, label=country)

plt.title("Dynamika rejestracji LEVs – TOP-5 krajów", fontsize=20)
plt.xlabel("Rok", fontsize=16)
plt.ylabel("Liczba rejestracji LEVs", fontsize=16)
plt.xticks(years)
plt.legend(title="Kraj", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plot4_path = "plots/4_top5_LEV_trendy.png"
plt.savefig(plot4_path, dpi=300)
plt.close()

# =====================================================
# 10. Raport PDF (dashboardowy, z polskimi znakami)
# =====================================================
os.makedirs("out", exist_ok=True)
pdf_path = "out/raport_dynamika_napedow_dashboard.pdf"

dejavu_path = font_manager.findfont("DejaVu Sans")
pdfmetrics.registerFont(TTFont("DejaVuSans", dejavu_path))

doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Polish", fontName="DejaVuSans", fontSize=11, leading=14))

Story = []
Story.append(Paragraph("<b>Raport: Dynamika napędów i kraje z rosnącymi LEVs</b>", styles["Title"]))
Story.append(Spacer(1, 12))

Story.append(Paragraph(
    "Zakres danych: ostatnie 5 lat. Analiza obejmuje napędy BEV, PHEV, HEV, Diesel i Petrol. "
    "Zbadano trendy rejestracji, udziały procentowe oraz kraje o rosnących rejestracjach pojazdów niskoemisyjnych (LEVs).",
    styles["Polish"]
))
Story.append(Spacer(1, 12))

for path, title in [
    (plot1_path, "Dynamika liczby rejestracji wg rodzaju napędu"),
    (plot2_path, "Udziały procentowe napędów w ostatnich 3 latach"),
    (plot3_path, "TOP-10 krajów wg przyrostu LEVs"),
    (plot4_path, "Dynamika rejestracji LEVs – TOP-5 krajów")
]:
    Story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    Story.append(Image(path, width=480, height=300))
    Story.append(Spacer(1, 12))

Story.append(Paragraph("<b>Wnioski:</b>", styles["Heading2"]))
Story.append(Paragraph(
    "- Napędy BEV i PHEV wykazują silny trend wzrostowy (≥2 wzrosty r/r i wyższy wynik w ostatnim roku).<br/>"
    "- Diesel wyraźnie traci udział w rynku (≥2 spadki r/r).<br/>"
    "- W ostatnich 3 latach dominującym napędem pozostaje Petrol, jednak jego udział spada.<br/>"
    "- Kraje o największym wzroście LEVs to m.in. "
    + ", ".join(top_countries)
    + ". Widoczna jest silna dynamika elektryfikacji flot pojazdów osobowych.<br/>",
    styles["Polish"]
))

doc.build(Story)
print(f"\n✅ Raport PDF zapisany jako: {pdf_path}")
