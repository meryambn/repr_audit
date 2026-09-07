import json

with open("benchmark_results/benchmark_data.json") as f:
    data = json.load(f)["models"]

print(
    f"{'Model':20s} | {'Anisotropy':10s} | {'SOD':4s} | {'CKA_Elbow':9s} | {'MaxDrift':8s} | {'AvgPlat':7s} | {'MAE(SOD)':8s} | {'MAE(CKA)':8s}"
)
print("-" * 90)
for k, v in data.items():
    cmp = v["probing_vs_geometry_comparison"]
    print(
        f"{k:20s} | {v['avg_anisotropy']:10.4f} | {v['sod_layer']:4d} | {v['cka_elbow_layer']:9d} | {v['max_drift_layer']:8d} | {cmp['average_probe_plateau']:7.2f} | {cmp['mae_sod_vs_probes']:8.2f} | {cmp['mae_cka_vs_probes']:8.2f}"
    )
