# 日前价格预测

import pandas as pd
from config import *

def forecast_da_price(df, target_date):
    hist = df[df[COL_DATE] < target_date]

    da_hourly = (
        hist.groupby("hour")[COL_DA_PRICE]
            .mean()
            .to_dict()
    )

    return da_hourly
