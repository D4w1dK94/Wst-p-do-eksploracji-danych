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
