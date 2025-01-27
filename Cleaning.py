import pandas as pd
import numpy as np

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
#Filling in missing account creation date with first active date
df_all['date_account_created'] = df_all['date_account_created'].fillna(df_all['timestamp_first_active'])
# Remove date_first_booking column(We're trying to predict country of first booking therefore it's useless)
df_all.drop('date_first_booking', axis=1, inplace=True)


# Remove outliers function
def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.nan, col_values)
    return df

#Removing outliers in age column and replace missing values with -1
df_all = remove_outliers(df_all, 'age', 15, 90)
df_all['age'] = df_all['age'].fillna(-1)

# Fill first_affiliate_tracked column
print("Filling first_affiliate_tracked column...")
df_all['first_affiliate_tracked'] = df_all['first_affiliate_tracked'].fillna(-1)
