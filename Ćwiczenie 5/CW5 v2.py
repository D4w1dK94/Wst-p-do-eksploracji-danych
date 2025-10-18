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

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import numpy as np

# =====================================================
# 1. Wczytanie danych
# =====================================================
df = pd.read_csv('../data/road_eqr_carpda.csv',
                 sep=None,
                 engine='python',
                 compression='infer')

print("Kolumny:")
print(list(df.columns))
print("\nPierwsze 10 wierszy:")
print(df.head(10))

# Utwórz folder 'out'
os.makedirs('out', exist_ok=True)

# Zapis podglądu
df.head(5000).to_excel('out/preview_original.xlsx', index=False)

# =====================================================
# 2. Redukcja kolumn
# =====================================================
keep_cols = ['Motor energy', 'Geopolitical entity (reporting)', 'TIME_PERIOD', 'OBS_VALUE']
keep_cols = [c for c in keep_cols if c in df.columns]
df_clean = df[keep_cols].copy()

df_clean.to_excel('out/df_clean.xlsx', index=False, sheet_name='dane')
df_clean.to_csv('out/df_clean.csv', index=False, encoding='utf-8')

# =====================================================
# 3. Wczytanie oczyszczonego pliku
# =====================================================
plik = 'out/df_clean.csv'
df = pd.read_csv(plik)

print("Kolumny po czyszczeniu:")
print(list(df.columns))
print("\nPróbka danych po czyszczeniu:")
print(df.head())

# Konwersja TIME_PERIOD na datetime
if 'TIME_PERIOD' in df.columns:
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y', errors='coerce')

# Typy kolumn
print("\nTypy kolumn (po konwersji):")
print(df.dtypes)

# =====================================================
# 4. Filtracja danych
# =====================================================
mask_ue = df['Geopolitical entity (reporting)'].astype(str).str.contains(
    r'European Union|EU27|EU28|EA19', case=False, na=False
)
mask_total = df['Motor energy'].astype(str).str.contains(r'Total|All', case=False, na=False)
df_filt = df[~mask_ue & ~mask_total].copy()
print(f"Liczba rekordów po filtracji: {len(df_filt)}")

# =====================================================
# 5. Ostatni rok
# =====================================================
if pd.api.types.is_datetime64_any_dtype(df_filt.get('TIME_PERIOD')):
    last_year = df_filt['TIME_PERIOD'].max()
    last_year_label = last_year.year
else:
    last_year = df_filt['TIME_PERIOD'].max()
    last_year_label = last_year

df_last = df_filt[df_filt['TIME_PERIOD'] == last_year].copy()
print(f"Ostatni dostępny rok: {last_year_label}, rekordy: {len(df_last)}")

# =====================================================
# 6. Sortowanie i konwersja OBS_VALUE
# =====================================================
df_filt['OBS_VALUE'] = pd.to_numeric(df_filt['OBS_VALUE'], errors='coerce')
df_last['OBS_VALUE'] = pd.to_numeric(df_last['OBS_VALUE'], errors='coerce')
df_last_sorted = df_last.sort_values(by=['OBS_VALUE', 'Geopolitical entity (reporting)'],
                                     ascending=[False, True])

# Zaokrąglanie do pełnej liczby
df_last_sorted['OBS_VALUE'] = df_last_sorted['OBS_VALUE'].apply(lambda x: int(round(x)))

# =====================================================
# 7. Statystyki opisowe
# =====================================================
print("\n=== Statystyki pełnego zbioru ===")
print(df_filt.describe(include='all'))

print("\n=== Statystyki dla ostatniego roku ===")
print(df_last_sorted.describe(include='all'))

# =====================================================
# 8. Wizualizacje dashboardowe
# =====================================================
os.makedirs('plots', exist_ok=True)

# Polskie znaki i style
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("talk")

# ---------- 8a. Top-10 krajów ----------
plt.figure(figsize=(12, 7))
top10 = df_last_sorted.groupby('Geopolitical entity (reporting)')['OBS_VALUE'].sum().sort_values(ascending=False).head(10)
colors = sns.color_palette("tab10", len(top10))
bars = plt.bar(top10.index, top10.values, color=colors)

plt.title(f"Top-10 krajów wg rejestracji pojazdów ({last_year_label})", fontsize=20)
plt.xlabel("Kraj", fontsize=16)
plt.ylabel("Liczba rejestracji", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + max(top10.values)*0.01, f"{int(h):,}", ha='center', va='bottom', fontsize=11)

plt.legend(bars, top10.index, title="Kraje", bbox_to_anchor=(1.05,1), loc='upper left', fontsize=12)
plt.tight_layout()
bar_path = f"plots/bar_last_year_top10_{last_year_label}_dashboard.png"
plt.savefig(bar_path, dpi=300)
plt.close()

# ---------- 8b. Poprawiony boxplot wg rodzaju napędu ----------
plt.figure(figsize=(14, 8))
order = df_last_sorted.groupby('Motor energy')['OBS_VALUE'].sum().sort_values(ascending=False).index

sns.boxplot(
    data=df_last_sorted,
    x='Motor energy',
    y='OBS_VALUE',
    palette="Set2",
    order=order,
    linewidth=1.5,   # cieńszy kontur, widać kolor
    fliersize=6,
    width=0.6,
    notch=False
)

plt.title(f"Rozkład liczby rejestracji wg rodzaju napędu ({last_year_label})", fontsize=20)
plt.xlabel("Rodzaj napędu", fontsize=16)
plt.ylabel("Liczba rejestracji", fontsize=16)
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)

handles = [plt.Rectangle((0,0),1,1,color=c) for c in sns.color_palette("Set2", len(order))]
plt.legend(handles, order, title="Rodzaj napędu", bbox_to_anchor=(1.05,1), loc='upper left', fontsize=12)

plt.tight_layout()
box_path = f"plots/box_last_year_by_fuel_{last_year_label}_dashboard_v3.png"
plt.savefig(box_path, dpi=300)
plt.close()

# ---------- 8c. Poprawiony wykres udziału procentowego ----------
plt.figure(figsize=(12, 7))
share = df_last_sorted.groupby('Motor energy')['OBS_VALUE'].sum().sort_values(ascending=False)
share_percent = 100 * share / share.sum()
share_percent = share_percent.round(1)
colors = sns.color_palette("Pastel1", len(share_percent))
bars = plt.bar(share_percent.index, share_percent.values, color=colors, width=0.6)

plt.title(f"Udział procentowy rejestracji wg rodzaju napędu ({last_year_label})", fontsize=20)
plt.xlabel("Rodzaj napędu", fontsize=16)
plt.ylabel("Udział [%]", fontsize=16)
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)

for bar, val in zip(bars, share_percent.values):
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 0.5,
        f"{val:.1f}%",
        ha='center', va='bottom', fontsize=11
    )

plt.legend(bars, share_percent.index, title="Rodzaj napędu", bbox_to_anchor=(1.05,1), loc='upper left', fontsize=12)
plt.tight_layout()
share_path = f"plots/share_by_fuel_percent_{last_year_label}_dashboard_v3.png"
plt.savefig(share_path, dpi=300)
plt.close()

# =====================================================
# 9. Raport PDF
# =====================================================
os.makedirs('out', exist_ok=True)
pdf_path = f"out/raport_analiza_{last_year_label}_dashboard_final.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
Story = []

Story.append(Paragraph(f"<b>Raport z analizy danych rejestracji pojazdów ({last_year_label})</b>", styles['Title']))
Story.append(Spacer(1,12))
Story.append(Paragraph("Filtr zastosowany: usunięto agregaty UE (European Union, EU27_2020, EU28, EA19) oraz sumy 'Total/All'.", styles['Normal']))
Story.append(Spacer(1,12))
Story.append(Paragraph(f"Ostatni dostępny rok: <b>{last_year_label}</b>", styles['Normal']))
Story.append(Spacer(1,12))

Story.append(Paragraph("<b>Wykresy:</b>", styles['Heading2']))
for img_path in [bar_path, box_path, share_path]:
    if os.path.exists(img_path):
        Story.append(Image(img_path, width=450, height=300))
        Story.append(Spacer(1,12))

Story.append(Paragraph("<b>Wnioski (przykładowe):</b>", styles['Heading2']))
Story.append(Paragraph(
    "- Największą liczbę rejestracji odnotowano w kilku największych krajach UE.\n"
    "- Napęd [tu wstaw własną obserwację] charakteryzuje się najwyższą medianą.\n"
    "- Rozkład napędów pokazuje dominację silników tradycyjnych, ale udziały elektrycznych rosną.\n"
    "- Zmienność między krajami jest duża, co sugeruje różny poziom rozwoju rynku pojazdów ekologicznych.",
    styles['Normal']
))

doc.build(Story)
print(f"\nRaport PDF zapisany do: {pdf_path}")
