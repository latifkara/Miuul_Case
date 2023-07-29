# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.

import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

df = sns.load_dataset('titanic')
df.head()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()
male, female = df["sex"].value_counts()
male
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

df.nunique()

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
print(df['pclass'].nunique())

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
print(df[["pclass", "parch"]].nunique())

# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
print(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)

# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == 'C']

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != 'S']
df[~(df["embarked"] == 'S')]

# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"] < 30) & (df["sex"] == "female")]

# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 500) | (df["age"] > 70)]

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()
df.isna().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.
df.drop('who', axis=1)

# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
max_repeat_value = df['deck'].value_counts().index[0]
df['deck'].mode()[0]
df['deck'].fillna(max_repeat_value)

# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].fillna(df["age"].median(), inplace=True)

# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(["pclass","sex"]).agg({"survived": ["sum", "count", "mean"]})

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
# setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)


def set_flag(x):
    if x >= 30:
        return 0
    else:
        return 1

age_flag = df["age"].apply(lambda x: set_flag(x))
df['age_flag'] = age_flag
df.head()
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

tips_df = sns.load_dataset("tips")
tips_df.head()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
tips_df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
tips_df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
tips_df.loc[(tips_df["time"] == "Lunch") & (tips_df["sex"] == "Female"), ["total_bill", "tip", "day"]].groupby("day")['total_bill','tip'].agg(["sum", "min", "max", "mean"])
tips_df.loc[(tips_df["time"] == "Lunch") & (tips_df["sex"] == "Female"), ["total_bill", "tip", "day"]].groupby("day").agg({'total_bill': ["sum", "min", "max", "mean"],
                                                                                                                           'tip': ["sum", "min", "max", "mean"]})

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
tips_df.loc[(tips_df["size"] < 3) & (tips_df["total_bill"] > 10), 'total_bill'].mean()

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
tips_df['total_bill_tip_sum'] = tips_df["total_bill"] + tips_df["tip"]

# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
new_dataframe = tips_df.sort_values('total_bill_tip_sum', ascending=False).head(30)







