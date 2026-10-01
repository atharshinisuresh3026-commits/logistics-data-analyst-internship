"""Week 4: predict actual delivery hours, validate, tune, then optimise carrier allocation with LP."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

df = pd.read_csv("data/clean_shipments.csv", parse_dates=["order_date"])
num = ["distance_km", "weight_kg", "num_stops", "traffic_index", "planned_hours", "month"]
cat = ["origin_hub", "carrier", "vehicle_type", "priority", "weather"]
X, y = df[num + cat], df["actual_hours"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
          "Gradient Boosting": GradientBoostingRegressor(random_state=42)}
cv = KFold(5, shuffle=True, random_state=42); out = {"models": {}}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)])
    cvr2 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="r2").mean()
    pipe.fit(Xtr, ytr); p = pipe.predict(Xte)
    out["models"][name] = dict(MAE=round(mean_absolute_error(yte, p), 3), RMSE=round(mean_squared_error(yte, p) ** .5, 3),
                               R2=round(r2_score(yte, p), 3), CV_R2=round(cvr2, 3))
# baseline: planned hours as the prediction
out["baseline_planned"] = dict(MAE=round(mean_absolute_error(yte, Xte.planned_hours), 3), RMSE=round(mean_squared_error(yte, Xte.planned_hours) ** .5, 3), R2=round(r2_score(yte, Xte.planned_hours), 3))

# hyper-parameter tuning on Gradient Boosting
grid = GridSearchCV(Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=42))]),
    {"m__n_estimators": [150, 300], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3, 4]}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(Xtr, ytr); best = grid.best_estimator_; p = best.predict(Xte)
out["best_params"] = grid.best_params_
out["tuned"] = dict(MAE=round(mean_absolute_error(yte, p), 3), RMSE=round(mean_squared_error(yte, p) ** .5, 3), R2=round(r2_score(yte, p), 3))
resid = yte - p; out["within_1h"] = round(float((resid.abs() <= 1).mean() * 100), 1)

# feature importance
names = best.named_steps["pre"].get_feature_names_out(); imp = pd.Series(best.named_steps["m"].feature_importances_, names).sort_values()
imp.index = [i.split("__")[1] for i in imp.index]; out["top_features"] = imp.sort_values(ascending=False).head(6).round(3).to_dict()
fig, ax = plt.subplots(figsize=(7, 4.2)); imp.tail(10).plot.barh(ax=ax, color="#4c72b0"); ax.set(title="Top 10 Feature Importances (Tuned Gradient Boosting)"); fig.tight_layout(); fig.savefig("figures/fig7_importance.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].scatter(yte, p, s=6, alpha=.4); ax[0].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
ax[0].set(title="Predicted vs Actual Hours", xlabel="Actual", ylabel="Predicted"); ax[1].hist(resid, bins=40, color="#55a868"); ax[1].set(title="Residuals", xlabel="Actual - Predicted (h)")
fig.tight_layout(); fig.savefig("figures/fig8_pred_resid.png", dpi=150); plt.close()

# ---- Optimisation: re-allocate zone demand across carriers to MINIMISE expected late shipments,
# ---- without increasing total shipping cost, subject to carrier capacity limits.
carriers = ["SwiftLine", "RoadRunner", "CargoPlus"]; zones = sorted(df.destination_zone.unique())
cost = df.groupby(["destination_zone", "carrier"]).shipping_cost.mean().unstack()[carriers].values      # zone x carrier (avg cost/shipment)
late = df.groupby("carrier").is_delayed.mean()[carriers].values                                         # late probability per carrier
demand = df.groupby("destination_zone").size().values.astype(float)
cur_share = df.carrier.value_counts(normalize=True)[carriers].values
cap = np.array([0.50, 0.45, 0.35]) * demand.sum()                                                       # assumed carrier capacity
nz, nc = cost.shape
c = np.tile(late, nz)                                                                                    # objective: expected late shipments
A_eq = np.zeros((nz, nz * nc))
for i in range(nz): A_eq[i, i * nc:(i + 1) * nc] = 1
A_ub, b_ub = [], []
for j in range(nc):
    r = np.zeros(nz * nc); r[j::nc] = 1; A_ub.append(r); b_ub.append(cap[j])
base_cost = float((cost * (demand[:, None] * cur_share)).sum())
A_ub.append(cost.flatten()); b_ub.append(base_cost)                                                      # cost must not exceed today's
sol = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=demand, bounds=(0, None), method="highs")
x = sol.x.reshape(nz, nc); opt_cost = float((cost * x).sum())
out["optimisation"] = dict(status=sol.message, baseline_cost=round(base_cost), optimised_cost=round(opt_cost),
    cost_change_pct=round((opt_cost - base_cost) / base_cost * 100, 2),
    new_share_pct=dict(zip(carriers, (x.sum(0) / x.sum() * 100).round(1))), old_share_pct=dict(zip(carriers, (cur_share * 100).round(1))),
    old_late_pct=round(float((cur_share * late).sum() * 100), 1), new_late_pct=round(float((x.sum(0) * late).sum() / x.sum() * 100), 1),
    late_shipments_avoided=round(float(((cur_share * late).sum() - (x.sum(0) * late).sum() / x.sum()) * x.sum())))
fig, ax = plt.subplots(figsize=(7, 3.8)); w = .38; ix = np.arange(nc)
ax.bar(ix - w / 2, cur_share * 100, w, label="Current"); ax.bar(ix + w / 2, x.sum(0) / x.sum() * 100, w, label="Optimised")
ax.set_xticks(ix); ax.set_xticklabels(carriers); ax.set(ylabel="Share of shipments (%)", title="Carrier Allocation: Current vs Optimised"); ax.legend(); fig.tight_layout(); fig.savefig("figures/fig9_allocation.png", dpi=150); plt.close()
json.dump(out, open("data/model_results.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
