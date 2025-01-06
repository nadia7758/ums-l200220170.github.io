import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('UAS/cleaned_data_group.csv')
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print(f"Kolom numerik yang dipilih: {numeric_columns}")

if len(numeric_columns) > 0:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_columns])

    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
    scaled_df.to_csv('scaled_data.csv', index=False)

    print(scaled_data.head())
else:
    print("Tidak ditemukan.")