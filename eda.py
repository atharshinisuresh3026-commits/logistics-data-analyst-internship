"""Week 3: exploratory analysis + visualisations. Saves figures/ and data/eda_results.json"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
df = pd.read_csv("data/clean_shipments.csv", parse_dates=["order_date"])
res = {}

# central tendency / spread
res["summary"] = df[["distance_km", "weight_kg", "actual_hours", "delay_hours", "shipping_cost"]].agg(["mean", "median", "std", "skew"]).round(2).to_dict()
res["delay_rate"] = round(df.is_delayed.mean() * 100, 1)
res["delay_by_weather"] = (df.groupby("weather").is_delayed.mean() * 100).round(1).to_dict()
res["delay_by_carrier"] = (df.groupby("carrier").is_delayed.mean() * 100).round(1).to_dict()
res["delay_by_hub"] = (df.groupby("origin_hub").is_delayed.mean() * 100).round(1).to_dict()
res["hours_by_vehicle"] = df.groupby("vehicle_type").actual_hours.mean().round(2).to_dict()
res["cost_per_km_by_vehicle"] = df.groupby("vehicle_type").cost_per_km.mean().round(2).to_dict()
res["avg_delay_by_carrier"] = df.groupby("carrier").delay_hours.mean().round(2).to_dict()
res["avg_delay_by_weather"] = df.groupby("weather").delay_hours.mean().round(2).to_dict()
res["express_hours"] = df.groupby("priority").actual_hours.mean().round(2).to_dict()
corr = df[["distance_km", "weight_kg", "traffic_index", "num_stops", "actual_hours", "delay_hours", "shipping_cost"]].corr().round(2)
res["corr_hours"] = corr["actual_hours"].to_dict(); res["corr_cost"] = corr["shipping_cost"].to_dict()
res["monthly_peak"] = df.groupby("month").size().sort_values(ascending=False).head(3).to_dict()

fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(df.actual_hours, bins=35, kde=True, ax=ax)
ax.axvline(df.actual_hours.mean(), color="red", ls="--", label=f"Mean {df.actual_hours.mean():.2f} h"); ax.legend()
ax.set(title="Distribution of Actual Delivery Time", xlabel="Hours"); fig.tight_layout(); fig.savefig("figures/fig1_delivery_hist.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(data=df, x="carrier", y="delay_hours", hue="weather", hue_order=["Clear", "Rain", "Storm"], ax=ax)
ax.axhline(0, color="grey", lw=.8); ax.set(title="Delay vs Plan by Carrier and Weather", ylabel="Delay (hours)"); fig.tight_layout(); fig.savefig("figures/fig2_delay_box.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 5)); sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix of Key Variables"); fig.tight_layout(); fig.savefig("figures/fig3_corr.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4)); sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="actual_hours", hue="vehicle_type", alpha=.5, ax=ax)
ax.set(title="Distance vs Delivery Time by Vehicle Type", xlabel="Distance (km)", ylabel="Hours"); fig.tight_layout(); fig.savefig("figures/fig4_scatter.png", dpi=150); plt.close()

m = df.groupby("month").agg(shipments=("shipment_id", "count"), delay=("is_delayed", "mean"))
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(m.index, m.shipments, color="#4c72b0", alpha=.8); ax.set(xlabel="Month", ylabel="Shipments", title="Monthly Shipment Volume and Delay Rate")
ax2 = ax.twinx(); ax2.plot(m.index, m.delay * 100, color="crimson", marker="o"); ax2.set_ylabel("Delay rate (%)"); ax2.grid(False)
fig.tight_layout(); fig.savefig("figures/fig5_monthly.png", dpi=150); plt.close()

hz = df.pivot_table(index="origin_hub", columns="destination_zone", values="is_delayed", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(7, 3.6)); sns.heatmap(hz, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "% delayed"})
ax.set(title="Delay Rate (%) by Origin Hub and Destination Zone", xlabel="Destination zone", ylabel=""); fig.tight_layout(); fig.savefig("figures/fig6_hub_zone.png", dpi=150); plt.close()
res["worst_hub_zone"] = {f"{h}|Z{z}": round(v, 1) for (h, z), v in hz.stack().sort_values(ascending=False).head(3).items()}
json.dump(res, open("data/eda_results.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
