
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Helpers import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

test_df.info()

train_df.head()
train_df.info()

check_df(train_df)

# Train data sets
## Preprocessing

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)

for col in cat_cols:
    cat_summary(train_df, col, plot=True)

for col in num_cols:
    num_summary(train_df, col, plot=True)

## Outliers

for col in num_cols:
    check_and_replacement(train_df, col, 0.1, 0.9, is_replacement=True, plot=False)


## Rare Analyser

rare_analyser(train_df, "SalePrice", cat_cols)
train_df = rare_encoder(train_df, 0.01)
rare_analyser(train_df, "SalePrice", cat_cols)

## Missing Value

na_cols = missing_values_table(train_df[num_cols], True)
train_df[na_cols] = train_df[na_cols].apply(lambda x: x.fillna(x.median()), axis=1)

missing_vs_target(train_df, "SalePrice", na_cols)
train_df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(train_df)


num_cols = [col for col in num_cols if col not in "SalePrice"]

# ignored the steps
df = pd.get_dummies(train_df[cat_cols+num_cols], drop_first=True)
df_test = pd.get_dummies(test_df[cat_cols+num_cols], drop_first=True)
df.head()
df.shape

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df_test.columns)
df_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)
df.head()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_test = pd.DataFrame(imputer.fit_transform(df_test), columns=df_test.columns)
df.head()
df.shape

# Standard Scaler
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
train_df[num_cols].head()


X = train_df[num_cols]
y = train_df["SalePrice"]
X.head()
y.head()

## Test data sets

cat_cols, num_cols, cat_but_car = grab_col_names(test_df)

## Outliers
for col in num_cols:
    check_and_replacement(test_df, col, 0.1, 0.9, is_replacement=False, plot=True)

## Missed Value
na_cols = missing_values_table(test_df[num_cols], na_name=True)
test_df[na_cols] = test_df[na_cols].apply(lambda x: x.fillna(x.median()), axis=1)
len(num_cols)
missing_values_table(test_df[num_cols])
missing_values_table(train_df[num_cols])
## Standard Scaler
num_cols = [col for col in num_cols if col not in "SalePrice"]

scaler = StandardScaler()
test_df[num_cols] = scaler.fit_transform(test_df[num_cols])
test_df[num_cols].head()

X_test = test_df[num_cols]

X = train_df[num_cols]
y = train_df["SalePrice"]

tree_mode = DecisionTreeRegressor()
tree_mode.fit(X, y)

y_pred = tree_mode.predict(X_test)
salePrice = pd.DataFrame({"SalePrice": y_pred})
salePrice.head()
test_df = test_df.join(salePrice)
test_df.head()

y_test = salePrice.copy()
# MSE

















































