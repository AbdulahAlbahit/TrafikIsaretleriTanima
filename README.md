# Farklı Görüntü Koşullarında Trafik İşareti Tanıma Sistemi

Bu proje, bilgisayarlı görü ve derin öğrenme tekniklerini kullanarak zorlu çevre koşulları altında trafik işaretlerini tespit edip sınıflandırmayı amaçlayan bir yapay zeka çalışmasıdır.

---

## Proje Özeti

Otonom sürüş ve gelişmiş sürüş destek sistemleri (ADAS), gerçek dünya şartlarında gece/karanlık, sis, yağmur, hareket bulanıklığı veya parlamalar gibi olumsuz hava ve ışık koşullarıyla karşılaşır. Standart ve ideal görüntüler üzerinde eğitilen modeller, bu tür bozucu etkenler altında performans kaybı yaşayabilir.

Bu projede, veri seti üzerindeki görüntülere sentetik bozulmalar uygulanarak farklı çevre koşulları simüle edilecek ve geliştirilen derin öğrenme modellerinin bu zorlu şartlar altındaki kararlılığı (robustness) analiz edilecektir.

---

## Temel Amaçlar

- **Veri İşleme ve Artırma:** İdeal trafik işareti görsellerine gece efekti, bulanıklık, gürültü ve kontrast değişiklikleri gibi bozucu etkenlerin eklenmesi.
- **Model Eğitimi:** Derin öğrenme ve nesne tespiti/sınıflandırma algoritmaları kullanılarak modelin eğitilmesi.
- **Performans Değerlendirmesi:** İdeal koşullar ile bozulmuş görüntü koşullarındaki model başarımlarının doğruluk metrikleri üzerinden karşılaştırılması.

---

## Veri Seti

Çalışmada, alanında standart olarak kabul edilen **GTSRB (German Traffic Sign Recognition Benchmark)** trafik işareti veri seti kullanılmaktadır.

- **Veri Seti:** 43 farklı trafik işareti sınıfı ve 50.000'''den fazla görsel.
- **Kaynak:** [Kaggle - GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## Kullanılan Teknolojiler

- **Programlama Dili:** Python
- **Ortam:** Google Colab / Jupyter Notebook
- **Kütüphaneler:** OpenCV, PyTorch / TensorFlow, NumPy, Matplotlib, Albumentations

---

## Çalıştırma ve Kullanım

Proje aşamaları geliştikçe ilgili notebook ve kod dosyaları bu depoya eklenecektir.

1. Depoyu klonlayın:
   `ash
   git clone https://github.com/AbdulahAlbahit/TrafikIsaretleriTanima.git
   `
