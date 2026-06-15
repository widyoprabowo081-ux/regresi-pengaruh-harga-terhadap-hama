import os
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ==============================================================================
# 1. MEMBACA DATASET SECARA OTOMATIS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(current_dir, "pengaruh laus lahan padi puso dan harga pupuk terhadap fluktasi harga beras eceran di kabupaten parigi mauotng menggunakan regresi linear berganda.csv")

# Membaca file CSV dengan pengaturan titik (.) sebagai pemisah ribuan
df = pd.read_csv(file_name, sep=';', skiprows=2, thousands='.')

# ==============================================================================
# 2. PREPROCESSING / PEMBERSIHAN DATA KETAT
# ==============================================================================
# Menyeragamkan nama kolom
df.columns = ['No', 'Bulan_Tahun', 'Luas_Puso', 'Harga_Pupuk', 'Harga_BBM', 'Upah_Buruh', 'Harga_Beras']

# ---- SOLUSI ERROR MATPLOTLIB ----
# 1. Hapus baris jika kolom Bulan_Tahun bernilai kosong/NaN
df = df.dropna(subset=['Bulan_Tahun'])
# 2. Paksa tipe data kolom Bulan_Tahun menjadi string murni dan bersihkan spasi
df['Bulan_Tahun'] = df['Bulan_Tahun'].astype(str).str.strip()
# 3. Pastikan string 'nan' (bekas data kosong) atau teks kosong benar-benar dibuang
df = df[(df['Bulan_Tahun'] != 'nan') & (df['Bulan_Tahun'] != '')]

# Memastikan kolom target analisis bertipe numerik (angka)
kolom_analisis = ['Luas_Puso', 'Harga_Pupuk', 'Harga_Beras']
for kolom in kolom_analisis:
    df[kolom] = pd.to_numeric(df[kolom], errors='coerce')

# Mengeliminasi baris jika ada nilai numerik yang kosong
df = df.dropna(subset=kolom_analisis)

# ==============================================================================
# 3. MENENTUKAN VARIABEL DAN MEMBENTUK MODEL REGRESI
# ==============================================================================
# X1 = Luas Puso, X2 = Harga Pupuk
X = df[['Luas_Puso', 'Harga_Pupuk']]
# Y = Harga Beras Eceran
y = df['Harga_Beras']

# Membuat model menggunakan Scikit-Learn untuk keperluan prediksi cepat
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)
y_pred = model_sklearn.predict(X)  # Hasil prediksi harga beras

# Menampilkan output ringkas Scikit-Learn ke terminal
print("=" * 60)
print("HASIL MODEL REGRESI LINEAR BERGANDA (SCIKIT-LEARN)")
print("=" * 60)
print(f"Persamaan Regresi: Y = {model_sklearn.intercept_:.2f} + ({model_sklearn.coef_[0]:.4f} * X1) + ({model_sklearn.coef_[1]:.4f} * X2)")
print(f"R-squared (R2)   : {model_sklearn.score(X, y):.4f}\n")

# Membuat model menggunakan Statsmodels untuk uji statistik dan signifikansi (p-value)
print("=" * 60)
print("HASIL UJI STATISTIK LENGKAP (STATSMODELS)")
print("=" * 60)
X_stat = sm.add_constant(X)
model_stat = sm.OLS(y, X_stat).fit()
print(model_stat.summary())

# ==============================================================================
# 4. VISUALISASI GRAFIK TREN YANG TERATUR (TIME SERIES LINE PLOT)
# ==============================================================================
plt.figure(figsize=(15, 6))

# Plot Garis 1: Data Aktual Lapangan (Warna Biru)
plt.plot(df['Bulan_Tahun'], y, color='#1f77b4', marker='o', markersize=4, 
         linestyle='-', linewidth=2, label='Harga Beras Aktual (Data Lapangan)')

# Plot Garis 2: Hasil Prediksi Rumus Regresi (Warna Merah Putus-putus)
plt.plot(df['Bulan_Tahun'], y_pred, color='#d62728', marker='x', markersize=4, 
         linestyle='--', linewidth=2, label='Harga Beras Prediksi (Model Regresi)')

# Mengatur Keterangan Judul dan Sumbu Grafik
plt.title('Fluktuasi Harga Beras Eceran di Kabupaten Parigi Moutong\nPerbandingan Tren Data Aktual vs Prediksi Model', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Periode (Bulan - Tahun)', fontsize=12, labelpad=10)
plt.ylabel('Harga Beras (Rp/Kg)', fontsize=12, labelpad=10)

# Mengatur Sumbu X agar rapi: Label bulan dimunculkan setiap 6 bulan (per semester) saja
plt.xticks(ticks=range(0, len(df), 6), labels=df['Bulan_Tahun'].iloc[::6], rotation=45)

# Menambahkan garis bantu (grid) transparan dan legenda petunjuk
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')

# Menyelaraskan tata letak agar layout tidak terpotong
plt.tight_layout()

# Menampilkan grafik ke layar
plt.show()