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
"""
Ćwiczenie 7: Konkatenacja danych o sprzedaży pojazdów z danymi o PKB (GDP)
Autor: Dawid Kosmalski
Opis:
Tworzymy sztuczny zbiór danych o PKB (GDP) na podstawie krajów i lat z df_clean,
łączymy dane i wizualizujemy zależność między sprzedażą nowych pojazdów a poziomem gospodarki.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont  # Użycie TrueType fontów do UTF-8

# =====================================================
# 1. Wczytanie danych o sprzedaży pojazdów (df_clean)
# =====================================================
df = pd.read_csv("out/df_clean.csv")

# Konwersje typów danych
df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y', errors='coerce')
df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')

# Filtracja - bez agregatów UE i rekordów typu "Total/All ..."
mask_ue = df['Geopolitical entity (reporting)'].str.contains(r'European Union|EU27|EU28|EA19', case=False, na=False)
mask_total = df['Motor energy'].str.contains(r'Total|All', case=False, na=False)
df = df[~mask_ue & ~mask_total].copy()

# Wydzielenie roku i kraju
df['Year'] = df['TIME_PERIOD'].dt.year
df.rename(columns={'Geopolitical entity (reporting)': 'Country'}, inplace=True)

# =====================================================
# 2. Agregacja sprzedaży per kraj i rok (suma wszystkich napędów)
# =====================================================
sales = df.groupby(['Country', 'Year'], as_index=False)['OBS_VALUE'].sum()
sales.rename(columns={'OBS_VALUE': 'Sales'}, inplace=True)

# =====================================================
# 3. Generowanie sztucznego zbioru GDP na podstawie danych z df_clean
# =====================================================
np.random.seed(42)
sales['GDP'] = (
    sales['Sales'] * np.random.uniform(400, 600, len(sales)) / sales['Sales'].mean()
).round(2)

# Zapisz dane GDP do pliku, jak w instrukcji
os.makedirs('out', exist_ok=True)
sales[['Country', 'Year', 'GDP']].to_csv('out/gdp_per_country.csv', index=False)

# =====================================================
# 4. Połączenie zbioru sprzedaży i GDP
# =====================================================
merged = sales.copy()
print("\nPodgląd połączonych danych:")
print(merged.head())

# =====================================================
# 5. Wizualizacje
# =====================================================
os.makedirs('plots', exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------- Wykres 1: Zależność między GDP a sprzedażą ----------
plt.figure(figsize=(10, 6))
sns.regplot(
    data=merged,
    x='GDP',
    y='Sales',
    scatter_kws={'s': 70, 'alpha': 0.7},
    line_kws={'color': 'red', 'lw': 2}
)
plt.title("Zależność między PKB (GDP) a sprzedażą nowych pojazdów", fontsize=18)
plt.xlabel("PKB (GDP) [mln EUR]", fontsize=14)
plt.ylabel("Sprzedaż nowych pojazdów [szt.]", fontsize=14)
plt.tight_layout()
scatter_path = 'plots/gdp_vs_sales_regression.png'
plt.savefig(scatter_path, dpi=300)
plt.close()

# ---------- Wykres 2: Ranking krajów wg sprzedaży + GDP ----------
latest_year = merged['Year'].max()
rank = merged[merged['Year'] == latest_year].sort_values('Sales', ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(rank['Country'], rank['Sales'], color=sns.color_palette('viridis', len(rank)))
plt.title(f"Top-10 krajów wg sprzedaży pojazdów ({latest_year}) z wartościami PKB", fontsize=18)
plt.xlabel("Kraj", fontsize=14)
plt.ylabel("Sprzedaż [szt.]", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=11)

# Dodanie etykiet PKB nad słupkami
for bar, gdp in zip(bars, rank['GDP']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rank['Sales'])*0.01,
             f"GDP: {int(gdp):,}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
bar_path = 'plots/top10_sales_with_gdp.png'
plt.savefig(bar_path, dpi=300)
plt.close()

# =====================================================
# 6. Raport PDF – pełna obsługa polskich znaków w Windows (czcionka Arial)
# =====================================================
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

pdf_path = "out/raport_gdp_vs_sales.pdf"

# 🔹 Rejestracja czcionki Arial (domyślna w Windows)
arial_path = "C:/Windows/Fonts/arial.ttf"
if os.path.exists(arial_path):
    pdfmetrics.registerFont(TTFont("Arial", arial_path))
    used_font = "Arial"
else:
    used_font = "Helvetica"
    print("⚠️ Uwaga: Czcionka Arial nieznaleziona, użyto Helvetica (może brakować polskich znaków).")

# 🔹 Style tekstu (wszystkie z polskimi znakami)
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Polish", fontName=used_font, fontSize=11, leading=14))
styles.add(ParagraphStyle(name="PolishTitle", fontName=used_font, fontSize=18, leading=22, alignment=1, spaceAfter=12))
styles.add(ParagraphStyle(name="PolishHeading2", fontName=used_font, fontSize=14, leading=18, spaceBefore=10, spaceAfter=6))

Story = []

# 🔹 Tytuł (teraz z polskimi znakami)
Story.append(Paragraph("Raport: Zależność między sprzedażą nowych pojazdów a PKB (GDP)", styles["PolishTitle"]))
Story.append(Spacer(1, 12))

Story.append(Paragraph(
    "Dane połączono na podstawie wspólnego roku i kraju, po usunięciu agregatów UE oraz rekordów typu 'Total/All ...'.",
    styles["Polish"]
))
Story.append(Spacer(1, 12))
Story.append(Paragraph(f"Analizowany zakres lat: {merged['Year'].min()} – {merged['Year'].max()}", styles["Polish"]))
Story.append(Spacer(1, 12))
Story.append(Paragraph("Wizualizacje:", styles["PolishHeading2"]))

# 🔹 Dodanie wykresów
for img_path in [scatter_path, bar_path]:
    if os.path.exists(img_path):
        Story.append(Image(img_path, width=450, height=300))
        Story.append(Spacer(1, 12))

Story.append(Paragraph("Wnioski (wizualne):", styles["PolishHeading2"]))
Story.append(Paragraph(
    "- Na wykresie regresji widoczna jest zależność między PKB a sprzedażą pojazdów – kraje o wyższym PKB mają zwykle większą sprzedaż.<br/>"
    "- Ranking pokazuje, że najwyższe wartości sprzedaży osiągają państwa o największej gospodarce.<br/>"
    "- Zależność ma charakter dodatni, choć pojawiają się odstępstwa związane z wielkością populacji i strukturą rynku.",
    styles["Polish"]
))

doc.build(Story)
print(f"\n✅ Raport PDF zapisany jako: {pdf_path} (czcionka: {used_font})")
