"""Freeze the week-3 configuration from TRAIN-only results. Writes reports/week3_frozen_config.json.

Nothing here reads the test file. After this file exists, the configuration must not change before the final
evaluation (src/final_eval.py checks the hashes recorded here).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd

from src import config
from src.pipelines import MODELS
from src.week3_experiments import (FULL_FIT_BUDGET_S, RESULTS3, adopt_if_consistent, aggregate3, best_tune_config, combine_configs,
                                    decide, N_SPLITS, skipped_ids)

FROZEN = config.REPORTS / "week3_frozen_config.json"
RULES = ("Mỗi thành phần tùy chọn (cắt xe giá cực cao, gom danh mục hiếm + model_version, tổ hợp cả hai) chỉ thay thế cấu hình đơn giản hơn "
         "khi MAE trung bình thấp hơn VÀ thắng ít nhất 4/5 fold so với cấu hình đó. Siêu tham số: chọn MAE trung bình thấp nhất trong các cấu hình "
         "đã thử và KHẢ THI về chi phí (thời gian huấn luyện ước tính trên toàn train không quá 30 phút; có thể hơi lạc quan vì chọn tốt nhất "
         "trong nhiều ứng viên; tập test sẽ cho ước lượng không chệch). Mô hình được chọn: MAE trung bình CV thấp nhất.")


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(agg_row: pd.Series) -> dict:
    return {"config_id": agg_row["config_id"], "model": agg_row["model"], "encoder": agg_row["encoder"], "scaler": agg_row["scaler"],
            "log_target": bool(agg_row["log_target"]), "params": json.loads(agg_row["params"]),
            "extreme_q": None if pd.isna(agg_row["extreme_q"]) else float(agg_row["extreme_q"]),
            "feature_set": agg_row["feature_set"],
            "rare_min_count": None if pd.isna(agg_row["rare_min_count"]) else int(agg_row["rare_min_count"]),
            "target_smooth": agg_row["target_smooth"],
            "cv": {k: float(agg_row[k]) for k in ("mae_mean", "mae_std", "rmse_mean", "r2_mean", "medape_pct_mean")}}


def choose_final(res: pd.DataFrame) -> dict:
    """Per model: the tune-best configuration, replaced by the combined configuration only if it is consistently better."""
    agg = aggregate3(res)
    by_id = agg.set_index("config_id")
    decisions = decide(res)
    combined = {c["model"]: c["config_id"] for c in combine_configs(res)}
    out = {}
    for m in MODELS:
        tune_id = decisions[m]["tune_best"]
        final_id, info = tune_id, None
        if m in combined and combined[m] in by_id.index and by_id.loc[combined[m], "n_folds"] == N_SPLITS:
            final_id, info = adopt_if_consistent(res, tune_id, [combined[m]])
        default = agg[(agg["model"] == m) & (agg["stage"] == "tune") & (agg["params"] == "{}") & (~agg["log_target"].astype(bool))].iloc[0]
        out[m] = {"final": _entry(agg[agg["config_id"] == final_id].iloc[0]),
                  "default": _entry(default), "tune_best": _entry(best_tune_config(agg, m)), "decisions": decisions[m],
                  "combined_check": info, "uses_combined": bool(final_id != tune_id)}
    return out


def reference_configs(res: pd.DataFrame, models: dict) -> dict:
    """Best-by-CV-MAE configuration of a model when it was ruled out for cost (its estimated full-train fit time exceeds the
    budget). It is NOT eligible for selection; it is evaluated once, trained on the same 20k development subsample used in CV,
    to show what the budget gave up. Frozen together with everything else, before the test split is read."""
    agg = aggregate3(res)
    out = {}
    for m in MODELS:
        t = agg[(agg["model"] == m) & (agg["stage"] == "tune") & (agg["n_folds"] == N_SPLITS)]
        top = t.loc[t["mae_mean"].idxmin()]
        if top["est_full_fit_s"] > FULL_FIT_BUDGET_S and top["config_id"] != models[m]["tune_best"]["config_id"]:
            e = _entry(top)
            e.update(fit_rows="dev_subsample", est_full_fit_s=float(top["est_full_fit_s"]), fit_seconds_cv=float(top["fit_seconds_mean"]))
            out[m] = e
    return out


def freeze() -> dict:
    res = pd.read_csv(RESULTS3)
    models = choose_final(res)
    selected = min(MODELS, key=lambda m: models[m]["final"]["cv"]["mae_mean"])
    frozen = {"created_at": datetime.now().isoformat(timespec="seconds"), "rules": RULES, "models": models, "selected_model": selected,
              "references": reference_configs(res, models), "skipped_configs": skipped_ids(),
              "full_fit_budget_s": FULL_FIT_BUDGET_S,
              "train_md5": file_md5(config.DATA_PROCESSED / "train.parquet"), "week3_results_md5": file_md5(RESULTS3)}
    FROZEN.write_text(json.dumps(frozen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return frozen


if __name__ == "__main__":
    f = freeze()
    print("selected model:", f["selected_model"])
    for m, v in f["models"].items():
        e = v["final"]
        print(f"{m:6s} CV MAE {e['cv']['mae_mean']:,.0f} (default {v['default']['cv']['mae_mean']:,.0f}) | {e['encoder']} {e['scaler']} "
              f"log={e['log_target']} q={e['extreme_q']} {e['feature_set']} rare={e['rare_min_count']} {e['params']}")
