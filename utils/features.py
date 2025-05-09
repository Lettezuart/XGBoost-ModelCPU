import pandas as pd

def add_datetime(df):
    """
    Adds a 'datetime' column by combining year, month, day, hour, minute, and second,
    and removes the original columns.

    Args:
        df (pd.DataFrame): DataFrame with separate time columns.

    Returns:
        pd.DataFrame: DataFrame with the new 'datetime' column and without the original columns.
    """
    df = df.copy()  # Per no modificar l'original

    # Crear columna datetime
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Eliminar columnes originals
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    return df

def add_weekday(df):
    """
    Adds a column with the day of the week (0=Monday, 6=Sunday).
    Requires the 'datetime' column.
    """
    if 'datetime' not in df.columns:
        df = add_datetime(df)
    df['weekday'] = df['datetime'].dt.weekday
    return df

def add_time_of_day(df):
    """
    Adds a 'time_of_day' column with values: morning, afternoon, evening, night.
    """
    df = df.copy()
    hour = df['datetime'].dt.hour

    def categorize(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    df['time_of_day'] = hour.apply(categorize)
    return df

def sort_by_datetime(df):
    """
    Sorts by 'datetime'.
    """
    df = df.copy()
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_glucose_diff(df):
    """
    Adds a column with the difference in glucose levels compared to the previous reading.
    """
    df = df.copy()
    df['glucose_diff'] = df['glucose_level'].diff().fillna(0)

    return df

def add_glucose_stats(df, window=12):
    """
    Adds columns for the average ('glucose_avg') and maximum ('glucose_max') glucose levels
    in the last 'window' intervals (for example, 12 intervals for the last hour if the data is
    recorded every 5 minutes).
    """
    df = df.copy()  # Per no modificar l'original

    # Afegir estadístiques per la finestra de 30 minuts (6 intervals)
    df['glucose_avg_30min'] = df['glucose_level'].rolling(window=6, min_periods=1).mean()
    df['glucose_max_30min'] = df['glucose_level'].rolling(window=6, min_periods=1).max()

    # Afegir estadístiques per la finestra de 60 minuts (12 intervals)
    df['glucose_avg_60min'] = df['glucose_level'].rolling(window, min_periods=1).mean()
    df['glucose_max_60min'] = df['glucose_level'].rolling(window, min_periods=1).max()

    return df

def add_flags(df):
    """
    Adds binary columns to indicate whether a relevant event has been recorded.
    """
    df = df.copy()
    df['insulin_flag'] = ((df['basal'] > 0) | (df['bolus'] > 0)).astype(int)
    df['meal_flag'] = df['meal'].notnull().astype(int)
    df['exercise_flag'] = df['exercise'].notnull().astype(int)
    df['sleep_flag'] = df['sleep'].notnull().astype(int)
    df['stress_flag'] = df['stressors'].notnull().astype(int)
    df['work_flag'] = df['work'].notnull().astype(int)
    df['ill_flag'] = df['illness'].notnull().astype(int)
    df['hypo_flag'] = df['hypo_event'].notnull().astype(int)
    return df

def add_lag_features(df, lags=[1, 2, 3]):
    """
    Adds columns with passed glucose levels values (lags).
    """
    df = df.copy()
    for lag in lags:
        df[f'glucose_lag_{lag}'] = df['glucose_level'].shift(lag)
    df.bfill(inplace=True)  # Omple els nuls amb el següent valor
    return df


def add_all_features(df):
    """
    Applies all features.
    """
    df = df.copy()
    df = add_datetime(df)
    df = sort_by_datetime(df)
    df = add_weekday(df)
    df = add_time_of_day(df)
    df = add_glucose_diff(df)
    df = add_flags(df)
    df = add_lag_features(df)
    df = add_glucose_stats(df)  
    return df

