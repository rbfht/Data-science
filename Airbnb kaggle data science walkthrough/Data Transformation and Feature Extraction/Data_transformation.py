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

#Adding new data from sessions.csv
sessions = pd.read_csv("../data/sessions.csv", header=0, index_col=False)

#Determine primary device
print("Determining primary device...")
sessions_device = sessions[["user_id", "device_type", "secs_elapsed"]]
aggregated_lvl1 = sessions_device.groupby(["user_id", "device_type"], as_index=False, sort=False).sum()
idx = aggregated_lvl1.groupby(["user_id"], sort=False)["secs_elapsed"].transform(max) == aggregated_lvl1["secs_elapsed"]
df_primary = aggregated_lvl1.loc[idx, ["user_id", "device_type", "secs_elapsed"]].copy()
df_primary.rename(columns = {"device_type":"primary_device", "secs_elapsed":"primary_secs"}, inplace = True)
df_primary = convert_to_binary(df = df_primary, column = "primary_device")
df_primary.drop("primary_device", axis=1, inplace=True)

#Determine secondary device
print("Determining secondary device...")
remaining =aggregated_lvl1.drop(aggregated_lvl1.index[idx])
idx_2 = remaining.groupby(["user_id"], sort=False)["secs_elapsed"].transform(max) == remaining["secs_elapsed"]
df_secondary = remaining.loc[idx_2,["user_id", "device_type", "secs_elapsed"]].copy()
df_secondary.rename(columns = {"device_type":"secondary_device", "secs_elapsed":"secondary_secs"})
df_secondary = convert_to_binary(df=df_secondary, column="secondary_device")
df_secondary.drop("secondary_device", axis=1, inplae=True)


# Count occurrences of value in a column
def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()

    df_counts = df[[id_col, column_to_convert]]
    df_counts["count"]=1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()
    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    new_df = new_df.fillna(0)

    #Rename columns
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-",
                                                                                                               "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns={category: col_name}, inplace=True)

    return new_df

# Aggregate and combine actions taken columns
print("Aggregating actions taken...")
sessions_actions = sessions[["user_id", "action", "action_type", "action_detail"]].copy()
columns_to_convert = ["action", "action_type", "action_detail"]
session_actions = session_actions.fillna('not provided')
first = True

for column in columns_to_convert:
    print("Converting " + column + " column...")
    current_data = convert_to_counts(df=session_actions, id_col='user_id', column_to_convert=column)

    if first:
        first=False
        actions_data = current_data
    else:
        actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

#Merge device datasets
print("Combining results...")
device_data = pd.merge(df_primary, df_secondary, how="outer", on="user_id")
# Merge device and actions datasets
combined_results = pd.merge(device_data, actions_data, on="user_id", how="outer")
df_sessions = combined_results.fillna(0)
# Merge user and session datasets(Inner join to ensure that all users in the training set will have sessions data)
df_all = pd.merge(df_sessions, df_all, on="user_id", how="inner")


