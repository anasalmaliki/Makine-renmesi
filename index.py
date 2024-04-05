import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Metin dosyasından veri okuma
file_path = "veri-seti1.txt"
df = pd.read_csv(file_path, delimiter="\t")

X = df.iloc[:, :-1]  # Hedef değişken hariç tüm öznitelikler
y = df.iloc[:, -1]   # Hedef değişken (son sütun)

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normalizasyon işlemi
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
# Çoklu Doğrusal Regresyon Analizi
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_coef = linear_reg.coef_
linear_intercept = linear_reg.intercept_
linear_predictions = linear_reg.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)

print("Çoklu Doğrusal Regresyon Katsayıları:", linear_coef)
print("Çoklu Doğrusal Regresyon Kesme Noktası:", linear_intercept)
print("Çoklu Doğrusal Regresyon Ortalama Kare Hatası (MSE):", linear_mse)

# Multinominal Lojistik Regresyon Analizi
logistic_reg = LogisticRegression(multi_class='multinomial', max_iter=1000)
logistic_reg.fit(X_train, y_train)
logistic_coef = logistic_reg.coef_
logistic_intercept = logistic_reg.intercept_
logistic_predictions = logistic_reg.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nMultinominal Lojistik Regresyon Katsayıları:", logistic_coef)
print("Multinominal Lojistik Regresyon Kesme Noktası:", logistic_intercept)
print("Multinominal Lojistik Regresyon Doğruluk Oranı:", logistic_accuracy)


# Naive Bayes Sınıflandırıcısı
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Test seti üzerinde tahminleri alın
y_pred_test_nb = nb_classifier.predict(X_test)

# Konfüzyon Matrisi
conf_matrix_nb = confusion_matrix(y_test, y_pred_test_nb)

# Hassasiyet, Özgünlük, Doğruluk, F1 Skoru
tn, fp, fn, tp = conf_matrix_nb.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy_nb = accuracy_score(y_test, y_pred_test_nb)
f1_score_nb = classification_report(y_test, y_pred_test_nb, output_dict=True)['weighted avg']['f1-score']

print("\nNaive Bayes Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_nb)
print("\nNaive Bayes Sınıflandırıcı Test Seti için Hassasiyet (Sensitivity):", sensitivity)
print("Naive Bayes Sınıflandırıcı Test Seti için Özgünlük (Specificity):", specificity)
print("Naive Bayes Sınıflandırıcı Test Seti için Doğruluk (Accuracy):", accuracy_nb)
print("Naive Bayes Sınıflandırıcı Test Seti için F1 Skoru:", f1_score_nb)

# Eğitim seti üzerinde tahminleri alın
y_pred_train_nb = nb_classifier.predict(X_train)

# Eğitim seti için sınıflandırma raporu
print("\nNaive Bayes Sınıflandırıcı Eğitim Seti için Sınıflandırma Raporu:")
print(classification_report(y_train, y_pred_train_nb))

# Test seti üzerinde tahminleri alın
y_pred_test_nb = nb_classifier.predict(X_test)

# Test seti için sınıflandırma raporu
print("\nNaive Bayes Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_test_nb))

# Test seti için karışıklık matrisi
print("\nNaive Bayes Sınıflandırıcı Test Seti için Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_test_nb))

# PCA modeli oluşturma
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)
# LDA modeli oluşturma
lda = LinearDiscriminantAnalysis(n_components=min(X_train_normalized.shape[1], len(set(y_train)) - 1))
X_train_lda = lda.fit_transform(X_train_normalized, y_train)
X_test_lda = lda.transform(X_test_normalized)
# PCA bileşenlerinin görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('PCA Bileşenleri')
plt.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')
plt.colorbar(label='Class')

# LDA bileşenlerinin görselleştirilmesi
plt.subplot(1, 2, 2)
plt.title('LDA Bileşenleri')
plt.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')
plt.colorbar(label='Class')

plt.tight_layout()
plt.show()
# Karar Ağacı Sınıflandırma Modeli
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=True)
plt.show()
# Karar Ağacı Sınıflandırma Modeli (Budama ile)
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=True)
plt.show()

# Test verisi için kestirim
y_pred = dt_classifier.predict(X_test)

n_components = min(X.shape[1], len(y.unique()) - 1) 
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)


# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Karar Ağacı Sınıflandırma Doğruluk Oranı:", accuracy)
print("\nKarar Ağacı Sınıflandırma Karmaşıklık Matrisi:\n", conf_matrix)
print("\nKarar Ağacı Sınıflandırma Sınıflandırma Raporu:\n", class_report)

# Test verisi için kestirim
y_pred = dt_classifier.predict(X_test)

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Karar Ağacı Sınıflandırma Doğruluk Oranı:", accuracy)
print("\nKarar Ağacı Sınıflandırma Karmaşıklık Matrisi:\n", conf_matrix)
print("\nKarar Ağacı Sınıflandırma Sınıflandırma Raporu:\n", class_report)

# Hangi özniteliklerin en ayırt edici olduğunu raporlama

# PCA için özniteliklerin önem sırasını belirleme
pca_components = pd.DataFrame(pca.components_, columns=X.columns)
print("\nPCA için özniteliklerin önem sırası:")
print(pca_components)

# LDA için özniteliklerin önem sırasını belirleme
lda_components = pd.DataFrame(lda.scalings_.T, columns=X.columns)
print("\nLDA için özniteliklerin önem sırası:")
print(lda_components)
