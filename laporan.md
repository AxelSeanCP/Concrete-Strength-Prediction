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
