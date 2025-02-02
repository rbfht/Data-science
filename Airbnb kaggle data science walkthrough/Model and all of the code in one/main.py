import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# Import data
print("Reading in data...")
tr_filepath = "../data/train_users_2.csv"
df_train = pd.read_csv(tr_filepath, header=0, index_col=None)
te_filepath = "../data/test_users.csv"
df_test = pd.read_csv(te_filepath, header=0, index_col=None)

# Combine into one dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Change Dates to consistent format
print("Fixing timestamps...")
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['date_account_created'] = df_all['date_account_created'].fillna(df_all['timestamp_first_active'])
df_all['date_account_created'] = df_all['date_account_created'].fillna(df_all['timestamp_first_active'])
df_all.drop('date_first_booking', axis=1, inplace=True)

def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.nan, col_values)
    return df

df_all = remove_outliers(df_all, 'age', 15, 90)
df_all['age'] = df_all['age'].fillna(-1)

print("Filling first_affiliate_tracked column...")
df_all['first_affiliate_tracked'] = df_all['first_affiliate_tracked'].fillna(-1)

def convert_to_binary(df, column):
    categories = list(df[column].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column[:5] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column] == category), col_name] = 1
    return df

print("One Hot Encoding categorical data...")
columns = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
           'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
           'first_device_type', 'first_browser']

for column in columns:
    convert_to_binary(df=df_all, column=column)
    df_all.drop(column, axis=1, inplace=True)

print("Adding new fields...")
df_all['day_account_created'] = df_all['date_account_created'].dt.weekday
df_all['month_account_created'] = df_all['date_account_created'].dt.month
df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter
df_all['year_account_created'] = df_all['date_account_created'].dt.year
df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour
df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday
df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month
df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter
df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year
df_all['created_less_active'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days

columns_to_drop = ["date_account_created", "timestamp_first_active", "date_first_booking"]
for column in columns_to_drop:
    if column in df_all.columns:
        df_all.drop(column, axis=1, inplace=True)

sessions = pd.read_csv("../data/sessions.csv", header=0, index_col=False)

print("Determining primary device...")
sessions_device = sessions[["user_id", "device_type", "secs_elapsed"]]
aggregated_lvl1 = sessions_device.groupby(["user_id", "device_type"], as_index=False, sort=False).sum()
idx = aggregated_lvl1.groupby(["user_id"], sort=False)["secs_elapsed"].transform(max) == aggregated_lvl1["secs_elapsed"]
df_primary = aggregated_lvl1.loc[idx, ["user_id", "device_type", "secs_elapsed"]].copy()
df_primary.rename(columns={"device_type": "primary_device", "secs_elapsed": "primary_secs"}, inplace=True)
df_primary = convert_to_binary(df=df_primary, column="primary_device")
df_primary.drop("primary_device", axis=1, inplace=True)

print("Determining secondary device...")
remaining = aggregated_lvl1.drop(aggregated_lvl1.index[idx])
idx_2 = remaining.groupby(["user_id"], sort=False)["secs_elapsed"].transform(max) == remaining["secs_elapsed"]
df_secondary = remaining.loc[idx_2, ["user_id", "device_type", "secs_elapsed"]].copy()
df_secondary.rename(columns={"device_type": "secondary_device", "secs_elapsed": "secondary_secs"}, inplace=True)
df_secondary = convert_to_binary(df=df_secondary, column="secondary_device")
df_secondary.drop("secondary_device", axis=1, inplace=True)

def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()
    df_counts = df[[id_col, column_to_convert]]
    df_counts["count"] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()
    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    new_df = new_df.fillna(0)
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns={category: col_name}, inplace=True)
    return new_df

print("Aggregating actions taken...")
session_actions = sessions[["user_id", "action", "action_type", "action_detail"]].copy()
columns_to_convert = ["action", "action_type", "action_detail"]
session_actions = session_actions.fillna('not provided')
first = True

for column in columns_to_convert:
    print("Converting " + column + " column...")
    current_data = convert_to_counts(df=session_actions, id_col='user_id', column_to_convert=column)
    if first:
        actions_data = current_data
        first = False
    else:
        actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

print("Combining results...")
device_data = pd.merge(df_primary, df_secondary, how="outer", on="user_id")
combined_results = pd.merge(device_data, actions_data, on="user_id", how="outer")
df_sessions = combined_results.fillna(0)
df_sessions.rename(columns={"user_id":"id"}, inplace=True)

df_all = pd.merge(df_all, df_sessions, on="id", how="left")
df_all = df_all.fillna(0)

# Split back into train/test using original data dimensions
train_len = len(df_train)
X_train = df_all[df_all['id'].isin(df_train['id'])].copy()
y_train = df_train['country_destination']

le = LabelEncoder()
y = le.fit_transform(y_train)

X = X_train.drop(['id', 'country_destination'], axis=1)
# Grid Search
XGB_model = xgb.XGBClassifier(
    objective='multi:softprob',
    subsample=0.5,
    colsample_bytree=0.5,
    seed=0,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3], 'n_estimators': [25, 50]}
model = GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=1, refit=True, cv=3)
model.fit(X, y)

print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Prepare test data
X_test = df_all[~df_all['id'].isin(df_train['id'])].copy()
test_ids = X_test['id'].values
X_test = X_test.drop('id', axis=1).fillna(-1)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Make predictions
y_pred = model.predict_proba(X_test)

# Repeat each id 5 times to match the 5 predictions per id
ids = np.repeat(test_ids, 5)
cts = []
for i in range(len(X_test)):
    # Append the top 5 countries for each test instance
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

print("Outputting final results...")
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('./submission.csv', index=False)