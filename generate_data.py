"""Generate a realistic, deliberately 'dirty' synthetic logistics dataset (reproducible)."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 5000

hubs = ["Hub_North", "Hub_South", "Hub_East", "Hub_West"]
carriers = ["SwiftLine", "RoadRunner", "CargoPlus"]
vehicles = ["Bike", "Van", "Truck"]

order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
hub = rng.choice(hubs, N, p=[0.3, 0.25, 0.25, 0.2])
zone = rng.integers(1, 9, N)
carrier = rng.choice(carriers, N, p=[0.4, 0.35, 0.25])
vehicle = rng.choice(vehicles, N, p=[0.25, 0.5, 0.25])
priority = rng.choice(["Standard", "Express"], N, p=[0.8, 0.2])
weather = rng.choice(["Clear", "Rain", "Storm"], N, p=[0.7, 0.24, 0.06])
distance = np.round(rng.gamma(4, 18, N) + 5, 1)                      # km
weight = np.round(rng.gamma(2.2, 14, N) + 0.5, 1)                     # kg
volume = np.round(weight * rng.uniform(0.004, 0.012, N), 3)           # m3
stops = rng.integers(1, 9, N)
traffic = np.round(np.clip(rng.normal(5.5, 2, N), 1, 10), 1)

speed = np.select([vehicle == "Bike", vehicle == "Van", vehicle == "Truck"], [22, 38, 32])
carrier_eff = np.select([carrier == "SwiftLine", carrier == "RoadRunner", carrier == "CargoPlus"], [-0.3, 0.2, 0.6])
weather_eff = np.select([weather == "Clear", weather == "Rain", weather == "Storm"], [0, 0.8, 2.4])
planned = np.round(2.6 + distance / speed * 1.45 + stops * 0.3 + weather_eff * 0.0 + 0.5, 2)   # customer-promised window
actual = (1.5 + distance / speed * (1 + 0.07 * traffic) + stops * 0.28 + weight * 0.01
          + weather_eff + carrier_eff - (priority == "Express") * 0.6 + rng.normal(0, 0.6, N))
actual = np.round(np.clip(actual, 0.5, None), 2)
cost = np.round(40 + distance * np.select([vehicle == "Bike", vehicle == "Van", vehicle == "Truck"], [3.2, 5.5, 8.5])
                + weight * 1.4 + (priority == "Express") * 120 + rng.normal(0, 25, N), 2)

df = pd.DataFrame(dict(shipment_id=[f"SH{100000+i}" for i in range(N)], order_date=order_date, origin_hub=hub,
    destination_zone=zone, carrier=carrier, vehicle_type=vehicle, priority=priority, weather=weather,
    distance_km=distance, weight_kg=weight, volume_m3=volume, num_stops=stops, traffic_index=traffic,
    planned_hours=planned, actual_hours=actual, shipping_cost=cost))

# ---- inject data-quality problems ----
for col, frac in [("weight_kg", .06), ("traffic_index", .05), ("distance_km", .02), ("weather", .03)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan
idx = rng.choice(N, 40, replace=False); df.loc[idx, "distance_km"] *= 10          # unit-entry errors
idx = rng.choice(N, 25, replace=False); df.loc[idx, "actual_hours"] *= 6           # extreme delays
idx = rng.choice(N, 15, replace=False); df.loc[idx, "shipping_cost"] *= -1         # sign errors
idx = rng.choice(N, 300, replace=False); df.loc[idx, "carrier"] = df.loc[idx, "carrier"].str.lower()   # inconsistent case
idx = rng.choice(N, 150, replace=False); df.loc[idx, "origin_hub"] = df.loc[idx, "origin_hub"] + " "    # trailing spaces
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
idx = rng.choice(N, 200, replace=False); df.loc[idx, "order_date"] = pd.to_datetime(df.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)             # duplicates
df.to_csv("data/raw_shipments.csv", index=False)
print("raw shape", df.shape)
