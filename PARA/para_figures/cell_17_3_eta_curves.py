# 17.3 BAcc along the whole eta grid, both blends (cached runs only, no training) -> eta_curves.csv for Figure F3
# Run after sections 1-9, 14.1, 14.3, 15.1 and 17.1 on the same dataset.
import pandas as pd
curve_rows, curve_missing = [], 0
for label, tag, T in PROMPT_SETS:
    z0 = tau0 * f0_test @ T.T
    for k in SWEEP_SHOTS:
        for d in range(SWEEP_DRAWS):
            keys = [cache_key(T, tag, k, d, s) for s in range(SWEEP_SEEDS)]
            if not all(key in zS_cache for key in keys):
                curve_missing += 1
                continue
            zs = [zS_cache[key].to(device) for key in keys]
            for bname, fn in BLENDS.items():
                for e in ETA_GRID:
                    acc, f1, bacc = np.mean([metrics3(fn(z0, z, float(e)), y_test) for z in zs], axis=0)
                    curve_rows.append([DATASET, C, label, k, d, bname, float(e), acc, f1, bacc])
        print(f"[{DATASET}|{label}] K={k} done")
if curve_missing:
    print(f"WARNING: {curve_missing} runs not cached and skipped.")
curves = pd.DataFrame(curve_rows, columns=["dataset", "classes", "prompts", "K", "draw", "blend", "eta", "acc", "macro_f1", "bacc"])
CURVE_FILE = os.path.join(RESULTS_DIR, "eta_curves.csv")
curves.to_csv(CURVE_FILE, index=False)
print(f"saved {CURVE_FILE} ({len(curves)} rows)")
print(curves[curves.prompts == curves.prompts.iloc[0]].pivot_table(index="eta", columns=["K", "blend"], values="bacc").round(1).to_string())
