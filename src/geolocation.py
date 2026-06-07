import pandas as pd

def prepare_ip_df_bounds(ip_df: pd.DataFrame,
                          lower: str = "lower_bound_ip_address",
                          upper: str = "upper_bound_ip_address") -> pd.DataFrame:
    """Ensure IP range boundary columns are integer dtype and sorted by lower bound.

    Returns a new DataFrame sorted by `lower` with integer bounds.
    """
    ip_df = ip_df.copy()
    ip_df[lower] = ip_df[lower].astype('int64')
    ip_df[upper] = ip_df[upper].astype('int64')
    ip_df = ip_df.sort_values(lower).reset_index(drop=True)
    return ip_df


def map_ip_to_country(df: pd.DataFrame,
                      ip_df: pd.DataFrame,
                      ip_int_col: str = 'ip_int',
                      lower: str = "lower_bound_ip_address",
                      upper: str = "upper_bound_ip_address",
                      country_col: str = 'country') -> pd.DataFrame:
    """Map integer IP addresses in `df[ip_int_col]` to countries using `ip_df` ranges.

    This uses a vectorized merge_asof to find the candidate range for each IP
    and labels any IP not falling within a matched range as 'Unknown'.
    Returns a new DataFrame with `country_col` populated.
    """
    df = df.copy()

    if ip_int_col not in df.columns:
        df[ip_int_col] = df['ip_address'].astype('int64')

    ip_df_sorted = prepare_ip_df_bounds(ip_df, lower=lower, upper=upper)

    # merge_asof requires both frames sorted by the merge key
    left = df.reset_index().sort_values(ip_int_col)
    right = ip_df_sorted.sort_values(lower)

    merged = pd.merge_asof(left, right, left_on=ip_int_col, right_on=lower, direction='backward', suffixes=(None, '_ip'))

    # If the ip_int is greater than the matched upper bound, it's not a match
    def _resolve_country(row):
        if pd.isna(row.get(upper)):
            return 'Unknown'
        try:
            return row['country'] if row[ip_int_col] <= row[upper] else 'Unknown'
        except Exception:
            return 'Unknown'

    merged[country_col] = merged.apply(_resolve_country, axis=1)

    # restore original order
    merged = merged.set_index('index')
    df[country_col] = merged[country_col]
    return df


__all__ = [
    'prepare_ip_df_bounds',
    'map_ip_to_country',
]
