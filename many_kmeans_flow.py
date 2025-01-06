from metaflow import FlowSpec, step, Parameter
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk analisis kata teratas
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Mengubah backend menjadi TkAgg untuk mendukung interaktif
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

class ManyClustersFlow(FlowSpec):

    @step
    def start(self):
        # Membaca data yang sudah diskalakan
        self.data_file = "UAS/scaled_data.csv"
        self.data = pd.read_csv(self.data_file)
        print(f"Data berhasil dibaca dari {self.data_file}")
        self.next(self.text_clustering)

    @step
    def text_clustering(self):
        # Asumsi ada kolom 'text' yang berisi data teks
        if 'text' not in self.data.columns:
            print("Tidak ada kolom teks untuk analisis.")
            self.next(self.end)
            return
        
        # Transformasi teks menjadi fitur numerik menggunakan TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data['text'])
        
        # Pilih hanya kolom numerik untuk clustering
        numeric_data = self.data.select_dtypes(include=["float64", "int64"])

        self.kmeans_results = {}
        for k in [3, 4, 5]:
            kmeans_model = KMeans(n_clusters=k, random_state=42)
            self.data[f'kmeans_cluster_{k}'] = kmeans_model.fit_predict(numeric_data)
            self.kmeans_results[k] = kmeans_model

        # Menampilkan kata teratas untuk setiap cluster berdasarkan TF-IDF
        for k in [3, 4, 5]:
            print(f"\nKata teratas untuk KMeans Cluster {k}:")
            cluster_centers = self.kmeans_results[k].cluster_centers_
            terms = vectorizer.get_feature_names_out()
            
            for i, center in enumerate(cluster_centers):
                # Mendapatkan 3 kata teratas dalam cluster berdasarkan bobot TF-IDF
                top_terms_idx = center.argsort()[-3:][::-1]
                top_terms = [terms[idx] for idx in top_terms_idx]
                print(f"Cluster {i}: {', '.join(top_terms)}")

        self.next(self.agglomerative_clustering)

    @step
    def agglomerative_clustering(self):
        # Pilih hanya kolom numerik untuk clustering
        numeric_data = self.data.select_dtypes(include=["float64", "int64"])

        # Membuat cluster untuk 3, 4, dan 5 cluster
        for k in [3, 4, 5]:
            agg_model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            self.data[f'agg_cluster_{k}'] = agg_model.fit_predict(numeric_data)
        
        # Menampilkan kata teratas untuk setiap cluster berdasarkan TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data['text'])
        
        for k in [3, 4, 5]:
            print(f"\nKata teratas untuk Agglomerative Clustering Cluster {k}:")
            cluster_labels = self.data[f'agg_cluster_{k}']
            for i in range(k):
                # Mengambil indeks dari cluster yang relevan
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_terms = tfidf_matrix[cluster_indices].mean(axis=0).A1
                top_terms_idx = cluster_terms.argsort()[-3:][::-1]
                top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_idx]
                print(f"Cluster {i}: {', '.join(top_terms)}")

        # Visualisasi hasil clustering untuk Agglomerative
        self.visualize_clusters(numeric_data)
        self.next(self.end)

    def visualize_clusters(self, numeric_data):
        # Reduksi dimensi untuk visualisasi
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(numeric_data)

        # Membuat visualisasi untuk 3, 4, dan 5 cluster dengan Plotly
        for k in [3, 4, 5]:
            fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], 
                            color=self.data[f'agg_cluster_{k}'],
                            title=f'Agglomerative Clustering dengan {k} Cluster',
                            labels={'x': 'PCA 1', 'y': 'PCA 2'})
            
            # Simpan dan buka visualisasi interaktif di browser
            output_filename = f"UAS/agg_clustering_{k}_clusters.html"
            fig.write_html(output_filename)
            print(f"Visualisasi untuk {k} cluster disimpan di: {output_filename}")
            fig.show()

    @step
    def end(self):
        # Simpan hasil clustering ke file
        output_file = "UAS/clustered_data.csv"
        self.data.to_csv(output_file, index=False)
        print(f"Hasil clustering disimpan di: {output_file}")

if __name__ == "__main__":
    ManyClustersFlow()
