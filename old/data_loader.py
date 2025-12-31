# 数据读取与标准化
import pandas as pd
from config import *

def time_to_interval_hour(t):
    hh, mm = map(int, t.split(":"))

    # 24:00 作为最后一个区间的结束时刻
    if hh == 24 and mm == 0:
        interval = 96
    else:
        # 以“区间结束时刻”计数：00:15 -> 1
        interval = hh * 4 + mm // 15

    # hour 用 interval 反推，避免 1:00 被错误归到 hour=1
    hour = min((interval - 1) // 4, 23)
    return interval, hour


def normalize_yes_no(series):
    """
    将 '是' / '否' 转为 1 / 0
    """
    return (
        series
        .astype(str)
        .str.strip()
        .replace({"是": 1, "否": 0})
        .astype(int)
    )


def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # 日期
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    # interval / hour
    df["interval"], df["hour"] = zip(
        *df[COL_TIME].apply(time_to_interval_hour)
    )

    # 节假日标准化（这是你刚才报错的地方）
    df["is_holiday"] = normalize_yes_no(df[COL_IS_HOLIDAY])
    df["is_weekend"] = normalize_yes_no(df[COL_IS_WEEKEND])

    return df
