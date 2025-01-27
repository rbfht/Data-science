import pandas as pd


def convert_to_binary(df, column):
    categories = list(df[column.drop_duplicates()])

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-",
                                                                                                               "").lower()
        col_name = column[:5] + '_' + cat_name[:10]
        df[col_name]=0
        df.loc[(df[column] == category), col_name] = 1

    return df


# Import data
print("Reading in data...")
tr_filepath = "../data/train_users_2.csv"
df_train = pd.read_csv(tr_filepath, header=0, index_col=None)
te_filepath = "../data/test_users.csv"
df_test = pd.read_csv(te_filepath, header=0, index_col=None)

# Combine into one dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

#One hot encoding
print("One Hot Encoding categorical data...")
columns = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

for column in columns:
    convert_to_binary(df=df_all, column=column)
    df_all.drop(column, axis=1, inplace=True)

#Adding new date related fields
df_all["day_account_created"] = df_all["date_account_created"].dt.weekday
df_all["month_account_created"]=df_all["date_account_created"].dt.month
df_all["year_account_created"]=df_all["date_account_created"].dt.year
df_all["quarter_account_created"]=df_all["date_account_created"].dt.quarter
df_all["hour_first_active"]=df_all["timestamp_first_active"].dt.hour
df_all["day_first_active"]=df_all["timestamp_first_active"].dt.day
df_all["month_first_active"]=df_all["timestamp_first_active"].dt.month
df_all["year_first_active"]=df_all["timestamp_first_active"].dt.year
df_all["quarter_first_active"]=df_all["timestamp_first_active"].dt.quarter
df_all['days_first_activity_after_creation'] = (df_all['timestamp_first_active'] - df_all['date_account_created']).dt.days

# Drop unnecessary columns(features only)
columns_to_drop = ["date_account_created", "timestamp_first_active", "date_first_booking", "country_destination"]
for column in columns_to_drop:
    if column in df_all.columns:
     df_all.drop(column, axis=1, inplace=True)