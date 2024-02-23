# Laporan Proyek Machine Learning - Axel Sean Cahyono Putra
![image of concrete compressive strength](img/Compressive-Strength-Of_Concrete.jpg)
# Concrete Compressive Strength
## Domain Proyek
"Concrete Compressive Strength" merupakan sebuah judul dataset yang didapat dari UCI Machine Learning Repository. dataset ini cocok untuk pemula dalam bidang data science ataupun machine learning. dikarenakan fitur yang tidak banyak dan dapat mudah dipahami, dataset ini cocok untuk pembelajaran bagi pemula.

Berdasarkan [cor-tuf.com](https://cor-tuf.com/everything-you-need-to-know-about-concrete-strength/), Concrete Compressive Strength atau Kekuatan Tekan Beton merupakan satuan yang dipakai oleh insinyur untuk mengukur kekuatan suatu beton untuk menahan beban yang dapat mengurangi ukuran beton tersebut. Kekuatan tekan beton diuji dengan memecahkan spesimen beton silinder dalam mesin khusus yang dirancang untuk mengukur jenis kekuatan ini. 

Kekuatan tekan dari beton sangatlah penting karena merupakan kriteria utama untuk menentukan performa dari suatu bangunan atau struktur. Dalam bisnis konstruksi, tentu saja terdapat beberapa resiko dan salah satunya adalah resiko akan bangunan yang roboh. Salah satu cara untuk mencegah hal itu adalah dengan mengukur kekuatan tekan dari beton, karena beton merupakan material utama dalam proses konstruksi jadi jika terdapat kesalahan dalam pengukuran kekuatan tekan beton akan berdampak sangat besar

Oleh karena itu dataset ini dapat digunakan untuk memprediksi kekuatan tekan beton sebelum dapat digunakan untuk kebutuhan konstruksi. Dalam dataset terdapat data train dan data test yang berisi komponen komponen yang digunakan dalam campuran suatu beton. Dengan menggunakan Regresi kita bisa melakukan prediksi terhadap data komponen tersebut dan menentukan komponen mana yang berpengaruh dalam menentukan kekuatan tekan beton untuk mengurangi resiko akan bangunan yang tidak aman.

## Business Understanding
### Problem Statement
- Bagaimana cara prediksi kekuatan tekan beton berdasarkan data komponen yang digunakan dalam campuran suatu beton

### Goals
- Berhasil melakukan prediksi kekuatan tekan beton menggunakan model machine learning

### Solution Statement
- Menggunakan EDA untuk mengetahui sifat dari data dan mengetahui fitur yang berpengaruh terhadap Kekuatan Tekan Beton
- Menggunakan beberapa model machine learning untuk memprediksi kekuatan tekan beton berdasarkan data komponen yang diberikan. Model yang akan dipakai adalah model regresi. Kemudian model dengan error paling kecil yang akan dipilih, beberapa model yang akan digunakan adalah:
    1. K-Nearest Neighbors
    2. Suport Vector Regressor
    3. Random Forest
    4. XGBoost Regressor

## Data Understanding
Dataset Concrete Compressive Strength dapat didownload melalui [link ini](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength). Dalam dataset terdapat 9 variabel dengan 8 fitur dan 1 target. Deskripsi variabel:
> fitur yang ada di dalam dataset ini merupakan komponen yang digunakan dalam campuran beton
- Cement (komponen 1): jumlah semen dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Blast Furnace Slag (komponen 2): jumlah slag dari tanur tiup dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Fly Ash (komponen 3): jumlah abu terbang dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Water (komponen 4): jumlah air dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Superplasticizer (komponen 5): jumlah superplastikizer dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Coarse Aggregate (komponen 6): jumlah agregat kasar dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Fine Aggregate (komponen 7): jumlah agregat halus dalam campuran beton yang diukur dalam satuan kg dalam meter kubik
- Age (Umur): umur beton dalam hari (1-365)
- Concrete compressive strength (Kekuatan Tekan Beton): kekuatan tekan beton yang diukur dalam satuan MPa (megapascal), merupakan target variabel dalam dataset

### Info Data
Dalam dataset terdapat 1030 sampel  

![info data](img/data_info.png)  
  

Mengecek nilai yang hilang  

![feature description](img/desc_feat.png)  
Disini diasumsikan fitur yang terdapat nilai 0 artinya komponen tersebut tidak digunakan dalam proses pencampuran beton tersebut. Sehingga yang bernilai 0 tidak akan di drop

### Visualisasi Data
- Univariate Analysis

![univar_analysis](img/univariate_analysis.png)

Diperlihatkan jumlah tiap tiap komponen yang digunakan dalam proses pencampuran, terdapat nilai 0 yang banyak pada beberapa fitur, proyek dilanjutkan dengan asumsi bahwa komponen yang bernilai 0 tidak digunakan dalam proses pencampuran

- Multivariate Analysis

![corr_matrix](img/correlation_matrix.png)

Terlihat bahwa banyak fitur yang tidak berkorelasi dengan target variabel, namun dengan hal ini dapat disimpulkan bahwa dataset bersifat **non-linear**. Water tidak di-drop karena diyakini bahwa air juga merupakan komponen penting dalam proses pencampuran beton.

## Data Preparation
Tahapan yang dilakukan dalam Data Preparation:

- Split variabel fitur dan variabel target, supaya mesin dapat membedakan variabel mana yang perlu digunakan dalam pelatihan

    ```
    X = concrete_data.drop(['Concrete compressive strength'], axis=1)
    y = concrete_data['Concrete compressive strength']
    ```
- Split data menjadi train dan test

    ```
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    ```
- Standardization: proses mengubah data sehingga memiliki skala yang sama yaitu mean = 0 dan varians = 1, hal ini diperlukan algoritma machine learning memiliki performa lebih baik ketika data memiliki skala yang sama
    ```
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)

    index, columns = X_train.index, X_train.columns

    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, index=index, columns=columns)
    X_train.head()
    ```
    > perlu diperhatikan dalam standarisasi dataset dengan semua fitur numerical terjadi perubahan dari dataframe menjadi np array. dapat dicegah dengan mengubah lagi menjadi dataframe seperti yang dituliskan di kode

## Modelling
Algoritma yang digunakan:
1. **K-Nearest Neighbor**:
    * Kelebihan: Mudah dipahami dan diimplementasikan, tidak memerlukan pembelajaran atau training yang kompleks.
    * Kelemahan: Kinerjanya lambat untuk dataset besar, sensitif terhadap data yang tidak terstandarisasi, dan perlu memilih parameter K yang tepat.
    * Parameter:
        - n_neighbors = jumlah tetangga terdekat yang akan digunakan untuk prediksi nilai target

2. **Support Vector Regressor (SVM)**:
    * Kelebihan: Efektif dalam dataset dengan banyak fitur, dapat menangani data non-linear melalui kernel, dan cenderung lebih toleran terhadap overfitting.
    * Kelemahan: Memerlukan tuning parameter yang tepat, seperti kernel dan C, serta tidak efisien untuk dataset sangat besar.
    * Parameter:
        - kernel = digunakan untuk mengubah data input ke dimensi yang lebih tinggi.
        - rbf = fungsi kernel yang berguna untuk data non-linear.

3. **Random Forest**
    * Kelebihan: Dapat menangani data yang tidak terstruktur dan fitur-fitur yang tidak terstandarisasi, tahan terhadap outliers dan noise, serta mudah digunakan.
    * Kelemahan: Kemungkinan overfitting pada dataset kecil dengan fitur-fitur yang sangat beragam, serta sulit untuk diinterpretasi.
    * Parameter:
        - n_estimators = jumlah pohon keputusan dalam random forest, makin banyak makin kompleks dan komputasinya mahal.
        - max_depth = kedalaman maksimum tiap pohon keputusan.
        - random_state = mengontrol randomness dalam model, jika diberi nilai maka akan dirandom secara konsisten.
        - n_jobs = jumlah pekerjaan yang akan digunakan secara paralel untuk pemrosesan.

4. **Extreme Gradient Boosting (XGBRegressor)**:
    * Kelebihan: Biasanya memberikan performa yang sangat baik, toleran terhadap overfitting, dan efisien dalam waktu komputasi.
    * Kelemahan: Memerlukan penyetelan parameter yang cermat, serta dapat memerlukan lebih banyak pemrosesan komputasi dibandingkan dengan model lainnya.
    * Parameter:
        - objective = fungsi tujuan untuk pemodelan. tujuannya adalah squared error regression loss yang cocok dengan loss function kita.
        - n_estimators = jumlah pohon keputusan dalam model ensemble.
        - max_depth = kedalaman maksimum tiap pohon keputusan.
        - learning_rate = mengontrol seberapa besar langkah pembelajaran yang diambil pada setiap iterasi.
        - subsample = Fraksi dari dataset yang akan digunakan untuk pelatihan setiap pohon.
        - colsample_bytree = Fraksi dari fitur yang akan digunakan dalam pembentukan setiap pohon. 

## Evaluation
Metrik evaluasi yang digunakan yaitu loss function **root_mean_squared_error (RMSE)**, implementasinya pada kode berikut

```
from sklearn.metrics import mean_squared_error

def rmse(y_pred, y_true):
  return np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
```

RMSE atau Root Mean Squared Error adalah loss function yang didapat dari proses mengkuadratkan error (y_asli - y_prediksi) dan dibagi jumlah yang menjadi rata-rata lalu di akarkan

Menggunakan metrik ini kita dapat melatih model dan mencari seberapa besar error yang didapat menggunakan formula:

![rmse](img/RMSE.jpg)

Dimana:  
RMSE = nilai root mean square error  
y  = nilai aktual  
ŷ  = nilai hasil prediksi  
i  = urutan data   
n  = jumlah data  

Berikut adalah jumlah loss dari tiap model

![model loss](img/model_loss_plot.png)

Terlihat bahwa model XGBRegressor memiliki jumlah loss paling kecil diantara ke-empat model. Sehingga model itulah yang terbaik dari model lainnya.

Berikut adalah hasil prediksi dari ke empat model

![hasil prediksi](img/model_prediction.png)