
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görev

## • Veri ön işleme,

df = pd.read_csv("datasets/hitters.csv")
df.head()


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


category_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("****************************************************")

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show(block=True)

for col in category_cols:
    cat_summary(df, col, plot=False)


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

for col in num_cols:
    num_summary(df, col, plot=False)

def target_summary_with_cat(dataframe, target, category_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(category_col)[target].mean()}), end="\n\n\n")

for col in category_cols:
    target_summary_with_cat(df, "Salary", col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)

stats.ttest_1samp(a=df["League"], b=df["NewLeague"])
df.head()
def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def check_outliers(dataframe, col_name, q1=0.25, q3=0.75):
    up_limit, low_limit = outlier_threshold(dataframe, col_name,  q1, q3)
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

def replace_with_thresholds(dataframe, variable):
    up_limit, low_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outliers(df, col))

for col in num_cols:
    sns.boxplot(data=df, x=col)
    plt.show(block=True)

for col in num_cols:
    print(col, check_outliers(df, col, 0.1, 0.9))

for col in num_cols:
    if check_outliers(df, col, 0.1, 0.9):
        replace_with_thresholds(df, col)



clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols].drop('Salary', axis=1))
df_score = clf.negative_outlier_factor_
df_score[:5]

np.sort(df_score)[:5]

scores = pd.DataFrame(np.sort(df_score))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_score)[2]

df[df_score < th]
df[df_score < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_score < th].index

df[df_score < th].drop(axis=0, labels=df[df_score < th].index)


def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_cols

na_columns = missing_values_table(df, na_name=True)

df.dropna(inplace=True)

df[category_cols].nunique()


df.loc[df["HmRun"].max(), 'League']
df.groupby(["HmRun", "League"]).agg({"Salary": "mean"}).sort_values(by="HmRUn", ascending=False)



# Label Encoder

def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in ['int64', 'float64'] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

# Rare encoder

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Salary", category_cols)

## Heatmap

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (10, 10)})
sns.heatmap(corr, cmap='RdBu')
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

## Özellik mühendisliği (Feature Engineering)

category_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "LStartDate"]

robust = RobustScaler()
df[num_cols] = robust.fit_transform(df[num_cols])
df.head()

# Split Data

X = df.drop(['Salary'], axis=1).values
y = df[["Salary"]].values
y.ndim

X.shape
y.shape

# Create Model
reg_model = LinearRegression().fit(X, y)

y_pred = reg_model.predict(X)
## MSE
mean_squared_error(y, y_pred)

## RMSE
np.sqrt(mean_squared_error(y, y_pred))

## MAE
mean_absolute_error(y, y_pred)

## Score
reg_model.score(X, y)

## train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

reg_model.score(X_train, y_train)

mean_squared_error(y_test, y_pred)

np.sqrt(mean_squared_error(y_test, y_pred))

np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))


# Create new Columns

df.info()
df["Years"].head()
df["Years"].describe()

df.head()
df["Years"] * 12
1987 - df["Years"]
df["LStartDate"] = df["Years"].apply(lambda x: 1987 - x)
df["Months"] = df["Years"].apply(lambda x: x * 12)

df["LStartDate"] = pd.to_datetime(df["LStartDate"], format="%Y")

df.head()

category_cols, num_cols, cat_but_car = grab_col_names(df)

target_summary_with_cat(df, "Salary", "LStartDate")

df.groupby(["Years", "LStartDate"])["Salary"].agg(["mean", "count"]).sort_values("mean", ascending=False)

df.groupby("Years")["CAtBat"].mean()

df["CAtBatPerYears"] = df["CAtBat"] / df["Years"]
df.shape



num_cols = [col for col in num_cols if col not in "LStartDate"]

robust = RobustScaler()
df[num_cols] = robust.fit_transform(df[num_cols])
df.head()

X = df.drop(['Salary', 'LStartDate'], axis=1).values
y = df[["Salary"]].values

missing_values_table(df)

df["CAtBatPerYears"].fillna(df["CAtBatPerYears"].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

reg_model.score(X_train, y_train)

mean_squared_error(y_test, y_pred)

np.sqrt(mean_squared_error(y_test, y_pred))

np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))





