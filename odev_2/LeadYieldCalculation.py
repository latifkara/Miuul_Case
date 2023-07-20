
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

# Görev 1: Aşağıdaki Soruları Yanıtlayınız

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
lead_df = pd.read_csv(r'F:\Miull_yaz_kampi\Miuul_Case\odev_2\datasets\persona.csv')
lead_df.head()
lead_df.info()

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

check_df(lead_df)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
lead_df["SOURCE"].nunique()
lead_df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
lead_df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
lead_df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
lead_df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
lead_df.groupby("COUNTRY")["PRICE"].sum()

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
lead_df.groupby("SOURCE")["PRICE"].count()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
lead_df.groupby("COUNTRY")["PRICE"].mean()

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
lead_df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
lead_df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
lead_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).round(2)


# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
agg_df = lead_df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).round(2).sort_values('PRICE',ascending=False)

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index()

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
agg_df.head()

# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

agg_df.info()
category_cols = [col for col in agg_df.columns if str(agg_df[col].dtype) in ["category", "object", "bool"]]

agg_df.head()
customers_level_based = agg_df[category_cols].apply(lambda x: '_'.join(x).upper(), axis=1)
customers_level_based = pd.DataFrame(customers_level_based, columns=["customers_level_based"])
customers_level_based = customers_level_based.join(agg_df["PRICE"])
customers_level_based.head()
agg_df = agg_df.join(customers_level_based["customers_level_based"])
agg_df.head()

# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
segment = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df["SEGMENT"] = segment
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
customers_level_based["customers_level_based"].unique()

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user].agg({"SEGMENT": "unique",
                                                         "PRICE": "mean"})
fra_user = "FRA_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == fra_user][["PRICE", "SEGMENT"]]




