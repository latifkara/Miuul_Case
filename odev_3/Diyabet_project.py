

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görev 1: Keşifçi Veri Analizi

# Adım 1: Genel resmi inceleyiniz.

df = pd.read_csv('dataset/diabetes.csv')
df.head()
df.info()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
       veri setindeki kategorik, numeric ve kategorik fakat kardinal değişkenlerin isimlerini verir.
       Not: kategorik değikenlerin içersine numerik görünümlü ketagorik değişkenler de dahildir.

       Args:
           dataframe: dataframe
               değişken isimleri alınmak istenen dataframe'dir,
           cat_th: int, float
               numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
           car_th: int, float
               kategorik fakat kardinal değişkenler için sınıf eşik değeri.

       Returns:
           cat_cols: list
               kategorik değişken listesi
           num_cols: list
               numerik değişken listesi
           cat_but_car: list
               kategorik görünümlü kardinal değişken listesi

       Notes:
           cat_cols + num_cols + cat_but_car = toplam değişken sayısı
           num_but_cat cat_cols'un içerisinde.

       """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                                                       dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                                                       dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #Number Columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_cols = [col for col in num_cols if dataframe[col].nunique() != len(dataframe)]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("****************************************************")

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"**********  {numerical_col} **********")
    print(dataframe[numerical_col].describe(quantiles).T)
    print("**********************************************")

    if plot:
        dataframe[numerical_col].hist()
        plt.title(numerical_col)
        plt.xlabel(numerical_col)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in num_cols:
    num_summary(df, col, plot=False)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

def target_summary_with_cat(dataframe, target, category_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(category_col)[target].mean()}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def check_outliers(dataframe, col_name):
    up_limit, low_limit = outlier_threshold(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)

def grab_outliers(dataframe, col_name, index=False):
    up_limit, low_limit = outlier_threshold(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].head())
    else:
        print(dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)])

    if index:
        outliers_index = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].index
        return outliers_index

for col in num_cols:
    print(col, ": ", check_outliers(df, col))

for col in num_cols:
    print(col)
    grab_outliers(df, col)

# Adım 6: Eksik gözlem analizi yapınız.

msno.bar(df)
plt.show()

def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_cols

missing_values_table(df)

# Adım 7: Korelasyon analizi yapınız.


corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (10, 10)})
sns.heatmap(corr, cmap='RdBu')
plt.show()

corr_matrix = df[num_cols].corr().abs()

upper_traingle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

sns.heatmap(upper_traingle_matrix , cmap='RdBu')
plt.show()


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
    corr = dataframe[num_cols].corr()
    corr_matrix = corr.abs()
    upper_traingle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_traingle_matrix.columns if any(upper_traingle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.heatmap(corr, cmap='RdBu')
        plt.show()

    return drop_list

high_correlated_cols(df, plot=True)

# Görev 2 : Feature Engineering

"""
Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
değerlere işlemleri uygulayabilirsiniz.

"""

df.loc[df["Glucose"] == 0, "Glucose"] = np.nan
df.loc[df["Insulin"] == 0, "Insulin"] = np.nan

df['Glucose'].isnull().sum()
df['Insulin'].isnull().sum()

missing_values_table(df)

df.groupby("Age")["Insulin"].median()
df.groupby("Age")["Glucose"].mean()
df["Glucose"].fillna(df.groupby("Age")["Glucose"].transform("mean"), inplace=True)


df.head()
na_cols = missing_values_table(df, na_name=True)
df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.mean()) if x.dtype  != "O" else x, axis=0)

missing_values_table(df)

# Adım 2: Yeni değişkenler oluşturunuz.

df.loc[(df["Age"] >= 21) & (df["Age"] < 51), "New_age_cat"] = "mature"
df.loc[(df["Age"] >= 51), "New_age_cat"] = "senior"

df.loc[(df["Age"] >= 51), "New_age_cat"]
df.head()

df['New_age_cat'].value_counts()

df.groupby("New_age_cat").agg({"Pregnancies": "mean"})
df.groupby("New_age_cat").agg({"Insulin": "mean"})
df.groupby("New_age_cat").agg({"Glucose": "mean"})
df.groupby("New_age_cat").agg({"BloodPressure": "mean"})

target_summary_with_num(df, "Outcome", "Insulin")
target_summary_with_num(df, "Outcome", "Glucose")

df["BMI"].unique()

df['New_age_labels'] = pd.cut(df["Age"], bins=[20, 24, 34, 44, 54, 65, df["Age"].max()], labels=['21-24', '25-34', '35-44', '45-54', '55-65', '+65'])
df['New_bmi_labels'] = pd.cut(df["BMI"], bins=[0.000, 20, 25, 30, 35, 45, 50, df["BMI"].max()], labels=['-20', '20-24.9', '25-29.9', '30-34.9', '35-44.9', '45-49.9', '+49.9'])
df.head()

missing_values_table(df)

df[df['New_bmi_labels'].isnull()]

df.loc[(df['New_bmi_labels'] == '-20'), 'New_weight_cat'] = "Underweight"
df.loc[(df['New_bmi_labels'] == '20-24.9'), 'New_weight_cat'] = "normal"
df.loc[(df['New_bmi_labels'] == '25-29.9'), 'New_weight_cat'] = "overweight"
df.loc[~((df['New_bmi_labels'] == '-20') | (df['New_bmi_labels'] == '20-24.9') | (df['New_bmi_labels'] == '25-29.9')), 'New_weight_cat'] = "obese"

df.loc[df["BMI"] == 0.00, 'New_bmi_labels'] = '-20'


for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

df.groupby(["New_age_cat", "New_weight_cat"]).agg({"Outcome": "mean"})
df.groupby(["New_age_cat", "New_weight_cat"]).agg({"Outcome": ["mean", "count"], "BloodPressure": "mean"})

df.head()

# Adım 3: Encoding işlemlerini gerçekleştiriniz.


def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in ['int64', 'float64'] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "Outcome", cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

one_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, one_cols)
df.head()
df.info()
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

# Adım 5: Model oluşturunuz.Adım 5: Model oluşturunuz.

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

