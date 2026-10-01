"""Week 2 pipeline: audit -> clean -> scale. Writes data/clean_shipments.csv and data/clean_log.json"""
import json, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("data/raw_shipments.csv")
log = {"raw_rows": len(df)}

# 1. Audit
log["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()
log["duplicates"] = int(df.duplicated().sum())
log["negative_cost"] = int((df.shipping_cost < 0).sum())

# 2. Structural fixes
df = df.drop_duplicates()
df["carrier"] = df["carrier"].str.strip().str.lower().map({"swiftline": "SwiftLine", "roadrunner": "RoadRunner", "cargoplus": "CargoPlus"})
df["origin_hub"] = df["origin_hub"].str.strip()
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=False, errors="coerce")
df["shipping_cost"] = df["shipping_cost"].abs()
log["after_dedup_rows"] = len(df)
log["bad_dates"] = int(df.order_date.isna().sum())

# 3. Outliers (IQR rule, grouped by vehicle type so a truck is not judged like a bike)
def iqr_flags(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)
for col in ["distance_km", "actual_hours"]:
    flag = df.groupby("vehicle_type")[col].transform(lambda s: iqr_flags(s))
    log[f"outliers_{col}"] = int(flag.sum())
    extreme = df[col] > df[col].quantile(.99) * 2          # clear data-entry errors
    log[f"extreme_{col}"] = int(extreme.sum())
    df.loc[extreme, col] = np.nan                           # treat as missing, then impute
    # winsorise remaining mild outliers
    lo, hi = df[col].quantile([.01, .99]); df[col] = df[col].clip(lo, hi)

# 4. Missing values
for col in ["weight_kg", "distance_km", "traffic_index", "actual_hours"]:
    df[col] = df[col].fillna(df.groupby("vehicle_type")[col].transform("median")).fillna(df[col].median())
df["weather"] = df["weather"].fillna(df["weather"].mode()[0])
log["missing_after"] = int(df.isna().sum().sum())

# 5. Feature engineering + scaling
df["delay_hours"] = (df.actual_hours - df.planned_hours).round(2)
df["is_delayed"] = (df.delay_hours > 0).astype(int)
df["month"] = df.order_date.dt.month
df["cost_per_km"] = (df.shipping_cost / df.distance_km).round(2)
num = ["distance_km", "weight_kg", "traffic_index", "num_stops"]
df[[c + "_minmax" for c in num]] = MinMaxScaler().fit_transform(df[num])
df[[c + "_z" for c in num]] = StandardScaler().fit_transform(df[num])
log["final_rows"] = len(df)
log["describe"] = df[["distance_km", "weight_kg", "actual_hours", "shipping_cost"]].describe().round(2).to_dict()
df.to_csv("data/clean_shipments.csv", index=False)
json.dump(log, open("data/clean_log.json", "w"), indent=1, default=str)
print(json.dumps(log, indent=1, default=str))
