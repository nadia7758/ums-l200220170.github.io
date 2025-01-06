import pandas as pd
import re

# Fungsi untuk membersihkan data
def bersihkan_teks(teks):
    if isinstance(teks, str):  # Periksa jika teks adalah string
        return re.sub(r'[^a-zA-Z0-9\s.,:;?!\'"()\-]', '', teks)
    return teks  # Kembalikan data jika bukan string

def main():
    # Path ke file CSV
    input_file = "UAS/data_group.csv"
    output_file = "UAS/cleaned_data_group.csv"
    
    try:
        # Membaca dataset dengan penanganan baris buruk
        df = pd.read_csv(input_file, on_bad_lines='skip')  # Abaikan baris yang bermasalah
        
        # Terapkan pembersihan pada semua kolom yang bertipe object (string)
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].apply(bersihkan_teks)
        
        # Simpan hasil pembersihan ke file baru
        df.to_csv(output_file, index=False)
        print(f"Data berhasil dibersihkan dan disimpan di: {output_file}")
    
    except FileNotFoundError:
        print(f"File {input_file} tidak ditemukan. Pastikan file berada di folder UAS.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
