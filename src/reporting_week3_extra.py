"""Write reports/week3_extra_report.md (boosting on the full train split, diagnostics, importance, learning curves).
Numbers come from the CSV/JSON results; interpretive claims are re-checked in verify_claims (Appendix A).

Run:  python -m src.reporting_week3_extra
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.reporting import md_table
from src.week3_extra import FRACTIONS, HGB_BEST, HGB_SPACE, N_HGB
from src.week3_extra_analysis import FEATURE_LABEL, NAME_LABEL, label

R, T = config.REPORTS, config.TABLES


def _t(name):
    return pd.read_csv(T / f"{name}.csv")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(pd.concat([a.rank(), b.rank()], axis=1).corr().iloc[0, 1])


# ------------------------------------------------------------------------------------------ claims
def verify_claims(c: dict) -> list[tuple[str, str, bool]]:
    out = []

    def add(i, t, ok):
        out.append((i, t, bool(ok)))

    cmp_ = c["cmp"].set_index("model")
    best = c["best"]
    d = best["default_same_encoder"]["cv"]["mae_mean"]
    f = best["final"]["cv"]["mae_mean"]
    add("X1", "Boosting đã tinh chỉnh có MAE CV thấp hơn boosting mặc định (cùng bộ mã hóa) trên toàn train", f < d)
    add("X2", "Boosting đã tinh chỉnh thắng KNN mặc định và Ridge mặc định ở ít nhất 4/5 fold (CV toàn train)",
        cmp_.loc["knn_default", "hgb_best_wins_folds"] >= 4 and cmp_.loc["ridge_default", "hgb_best_wins_folds"] >= 4)
    add("X3", "Boosting đã tinh chỉnh có MAE CV (toàn train) thấp hơn Random Forest mặc định", cmp_.loc["hgb_best", "mae_mean"] < cmp_.loc["rf_default", "mae_mean"])
    ds = c["diag"]
    dv = ds[ds["data"] == "dev20k"].set_index("name")
    add("X4", "Random Forest mặc định khớp train rất sát: MAE train nhỏ hơn một nửa MAE validation", dv.loc["rf_default", "mae_train"] < 0.5 * dv.loc["rf_default", "mae_valid"])
    add("X5", "Ridge ít overfit nhất: tỷ lệ MAE validation/train nhỏ nhất trong các mô hình", dv["valid_over_train_mae"].idxmin() == "ridge_tuned")
    imp = c["imp"]
    dd = imp[imp["data"] == "dev20k"]
    models = sorted(dd["name"].unique())
    ident = dd[dd["feature"].str.startswith("identity")].set_index("name")["delta_mae"]
    top_single = dd[dd["kind"] == "single"].groupby("name")["delta_mae"].max()
    add("X6", "Nhóm định danh xe (hãng+mẫu) làm tăng MAE nhiều hơn bất kỳ cột đơn lẻ nào, ở mọi mô hình", all(ident[m] > top_single[m] for m in models))
    mileage_rank = dd[dd["feature"] == "mileage_km"].set_index("name")["rank_in_model"]
    add("X7", "mileage_km nằm trong 5 cột quan trọng nhất (xét 15 cột đơn lẻ) ở ít nhất một nửa số mô hình", (mileage_rank <= 5).mean() >= 0.5)
    piv = dd[dd["kind"] == "single"].pivot(index="feature", columns="name", values="delta_mae")
    if {"rf_tuned", "hgb_best"} <= set(piv.columns):
        add("X8", "Thứ hạng độ quan trọng của Random Forest tinh chỉnh và boosting tương quan (Spearman ≥ 0,6)", _spearman(piv["rf_tuned"], piv["hgb_best"]) >= 0.6)
    cr = c["cross"].sort_values("fraction")
    small, full = cr.iloc[0], cr.iloc[-1]
    add("X9", "Ở phần dữ liệu nhỏ nhất Random Forest tinh chỉnh tốt hơn mặc định, nhưng ở 100% dữ liệu thì tệ hơn (đảo thứ hạng: giả thuyết giải thích kết quả test)",
        small["tuned_minus_default"] < 0 < full["tuned_minus_default"])
    cs = c["curves"]
    rfd = cs[cs["name"] == "rf_default"].sort_values("fraction")["mae_valid"].to_numpy()
    add("X10", "MAE validation của Random Forest mặc định giảm đơn điệu khi tăng dữ liệu huấn luyện", bool((np.diff(rfd) < 0).all()))
    return out


# ------------------------------------------------------------------------------------------ sections
def section_intro(c):
    return f"""# Báo cáo bổ sung Tuần 3: boosting, chẩn đoán overfit, độ quan trọng đặc trưng, đường cong học
{c['banner']}
_Mọi số liệu tính từ `reports/tables/week3_*.csv` (chỉ dùng **tập train**). Sinh bằng `python -m src.reporting_week3_extra`._

## 1. Mục tiêu và nguyên tắc

Báo cáo chính tuần 3 còn bốn khoảng trống: chưa có mô hình boosting, chưa có chẩn đoán overfit, chưa biết biến nào quyết định giá, và giải thích vì sao Random Forest tinh chỉnh thua mặc định trên test mới chỉ là giả thuyết. Phần bổ sung này lấp các khoảng trống đó.

**Nguyên tắc quan trọng:** tập test đã được đánh giá **một lần** (`src/final_eval.py`) với các mô hình đã đóng băng. Boosting được thêm *sau* đó nên **không được đánh giá trên tập test** (xem lần hai là biến test thành tập chọn mô hình). Mọi so sánh ở đây là cross-validation trên train; số liệu test của báo cáo chính không thay đổi. Vì boosting rất nhanh, nó được tinh chỉnh trên **toàn bộ train** ({c['n_train']:,} dòng) chứ không phải tập con 20k như các mô hình khác.
"""


def section_hgb(c):
    best, cmp_ = c["best"], c["cmp"].copy()
    enc = pd.DataFrame(best["encoder_stage"])
    enc["encoder"] = enc["encoder"].map({"ordinal": "Ordinal", "target": "Target", "onehot": "One-hot"})
    enc["log giá"] = enc["log_target"].map({True: "có", False: "không"})
    enc = enc[["encoder", "log giá", "mae_mean", "r2_mean", "medape_pct_mean"]].rename(columns={"mae_mean": "MAE CV", "r2_mean": "R² CV", "medape_pct_mean": "MedAPE %"})
    cmp_["model"] = cmp_["model"].map(lambda m: {"hgb_default": "Boosting (mặc định)", "hgb_best": "Boosting (tinh chỉnh)"}.get(m, label(m)))
    cmp_["boosting tinh chỉnh thắng (số fold)"] = cmp_["hgb_best_wins_folds"].map(lambda v: "" if pd.isna(v) else f"{int(v)} / 5")
    cmp_ = cmp_[["model", "source", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "medape_mean", "fit_seconds", "boosting tinh chỉnh thắng (số fold)"]]
    cmp_.columns = ["mô hình", "nguồn", "MAE", "MAE std (fold)", "RMSE", "R²", "MedAPE %", "fit (s)", "boosting tinh chỉnh thắng (số fold)"]
    fin = best["final"]
    space = "; ".join(f"{k} ∈ {{{', '.join(str(x) for x in v)}}}" for k, v in HGB_SPACE.items())
    vd = best["version_decision"]
    return f"""## 2. Boosting (HistGradientBoosting) tinh chỉnh trên toàn train

**Mô hình:** `HistGradientBoostingRegressor` của scikit-learn, số vòng lặp cố định (không dùng tách validation bên trong), đầu vào qua cùng pipeline (điền thiếu → mã hóa → mô hình).
**Quy trình:** (1) chọn bộ mã hóa × log giá với tham số mặc định; (2) tìm kiếm ngẫu nhiên {N_HGB} bộ siêu tham số ({space}); (3) thử thêm `model_version` (Target Encoding, gom danh mục hiếm) và chỉ nhận nếu thắng ≥ 4/5 fold. Tất cả bằng GroupKFold 5 fold trên toàn train.

**Bước 1 – bộ mã hóa với tham số mặc định:**

{md_table(enc)}

**Cấu hình boosting được chọn:** {fin['encoder']}, {'log giá' if fin['log_target'] else 'giá gốc'}, `{json.dumps(fin['params'])}`{', có model_version' if best['adopted_model_version'] else ', không dùng model_version'} (kiểm tra model_version: MAE {vd['baseline_mae']:,.0f} → ứng viên tốt nhất {vd['chosen_mae']:,.0f}; {'được nhận' if best['adopted_model_version'] else 'không đủ điều kiện 4/5 fold'}).

**So sánh trên toàn train (cùng các fold; các mô hình khác ở tham số mặc định, từ xác nhận tuần 2):**

{md_table(cmp_)}

![Boosting: cấu hình đã thử và so sánh](figures/week3_hgb.png)

**Lưu ý trung thực:** (a) boosting đã được tinh chỉnh còn các mô hình ở cột so sánh thì chưa (mặc định tuần 2), nên chênh lệch một phần phản ánh nỗ lực tinh chỉnh, không chỉ loại mô hình; (b) chọn tốt nhất trong {N_HGB} ứng viên làm MAE CV hơi lạc quan; (c) **chưa có số liệu test cho boosting**.
"""


def section_diag(c):
    d = c["diag"].copy()
    d = d[d["data"] == "dev20k"]
    t = pd.DataFrame({"mô hình": d["name"].map(label), "MAE train": d["mae_train"], "MAE validation": d["mae_valid"], "tỷ lệ valid/train": d["valid_over_train_mae"],
                      "R² train": d["r2_train"], "R² validation": d["r2_valid"], "MedAPE % validation": d["medape_valid"], "fit (s)": d["fit_seconds"]})
    full = c["diag"][c["diag"]["data"] == "full"]
    extra = ""
    if len(full):
        r = full.iloc[0]
        extra = (f"\nTrên các fold của **toàn train** ({int(r['n_folds'])} fold), boosting tinh chỉnh có MAE train {r['mae_train']:,.0f} và validation {r['mae_valid']:,.0f} "
                 f"(tỷ lệ {r['valid_over_train_mae']:.2f}).\n")
    return f"""## 3. Chẩn đoán overfit: train so với validation

Mỗi mô hình (cấu hình đóng băng, thêm Random Forest mặc định và boosting) được huấn luyện lại trên từng fold của CV (tập con 20k), rồi đo trên chính phần huấn luyện và trên fold validation.

{md_table(t)}
{extra}
![Train so với validation](figures/week3_diag_train_valid.png)

**Cách đọc:** {c['diag_text']}
"""


def section_importance(c):
    imp = c["imp"]
    dd = imp[imp["data"] == "dev20k"]
    rows = []
    for m in [n for n in NAME_LABEL if n in set(dd["name"])]:
        s = dd[(dd["name"] == m) & (dd["kind"] == "single")].sort_values("delta_mae", ascending=False).head(5)
        rows.append({"mô hình": label(m), "5 cột đơn lẻ quan trọng nhất (% tổng mức tăng MAE)": "; ".join(f"{r.feature} ({r.share_pct:.0f}%)" for r in s.itertuples())})
    j = dd[dd["kind"] == "joint"].copy()
    j["mô hình"] = j["name"].map(label)
    j["nhóm cột"] = j["feature"].map(lambda x: FEATURE_LABEL.get(x, x))
    jt = j.pivot(index="mô hình", columns="nhóm cột", values="delta_mae").reset_index()
    return f"""## 4. Độ quan trọng đặc trưng (permutation importance, trên validation)

**Phương pháp:** với mỗi fold, xáo trộn một cột gốc (hoặc một nhóm cột cùng thông tin) giữa các dòng validation và đo mức tăng MAE; lặp {3} lần, lấy trung bình qua các fold. Làm trên cột **gốc** (trước mã hóa), nên One-hot không làm loãng độ quan trọng của một biến. Các cột có thông tin chồng nhau (hãng, mẫu; năm sản xuất, tuổi xe) làm độ quan trọng từng cột bị chia nhỏ, nên có thêm hai phép xáo trộn **cả nhóm**: "định danh xe" (hãng + mẫu [+ phiên bản]) và "tuổi xe" (năm + tuổi). Chỉ dùng tập train (validation của CV), không dùng tập test.

**Năm cột đơn lẻ quan trọng nhất của từng mô hình:**

{md_table(pd.DataFrame(rows))}

**Mức tăng MAE (EUR) khi xáo trộn cả nhóm:**

{md_table(jt)}

![Độ quan trọng đặc trưng](figures/week3_importance.png)

**Lưu ý khi diễn giải:** độ quan trọng hoán vị cho biết mô hình *dựa vào* cột nào, không phải quan hệ nhân quả; cột tương quan với nhau chia sẻ độ quan trọng; giá trị là mức tăng sai số khi phá vỡ quan hệ giữa cột đó và các cột còn lại.
"""


def section_curves(c):
    cr = c["cross"].sort_values("fraction").copy()
    t = pd.DataFrame({"tỷ lệ dữ liệu huấn luyện": (cr["fraction"] * 100).round(0).astype(int).astype(str) + "%", "số dòng": cr["n_train_used"].round(0), "MAE RF mặc định": cr["rf_default_mae"],
                      "MAE RF tinh chỉnh": cr["rf_tuned_mae"], "tinh chỉnh − mặc định": cr["tuned_minus_default"], "fold tinh chỉnh tốt hơn": cr["folds_tuned_better"].astype(int).astype(str) + " / " + cr["n_folds"].astype(int).astype(str)})
    cs = c["curves"]
    big = cs[cs["fraction"] == cs["fraction"].max()].set_index("name")
    small = cs[cs["fraction"] == cs["fraction"].min()].set_index("name")
    gain = pd.DataFrame({"mô hình": [label(n) for n in big.index], f"MAE ở {FRACTIONS[0] * 100:.0f}% dữ liệu": small.loc[big.index, "mae_valid"].to_numpy(),
                         "MAE ở 100% dữ liệu": big["mae_valid"].to_numpy(), "giảm (%)": ((1 - big["mae_valid"] / small.loc[big.index, "mae_valid"]) * 100).to_numpy()})
    return f"""## 5. Đường cong học: vì sao Random Forest tinh chỉnh thua mặc định trên test?

**Giả thuyết cần kiểm chứng (từ báo cáo chính):** cấu hình Random Forest được chọn trên ~16.000 dòng huấn luyện (cây nông hơn, ít đặc trưng mỗi lần chia hơn) tốt ở dữ liệu nhỏ nhưng kém ở dữ liệu lớn, nơi cây sâu có lợi. **Cách kiểm chứng:** CV 5 fold trên toàn train, huấn luyện trên {', '.join(f'{f * 100:.0f}%' for f in FRACTIONS)} của phần huấn luyện (lấy theo nhóm xe nguyên vẹn), đo MAE trên fold validation.

{md_table(t)}

![Đường cong học](figures/week3_learning_curves.png)

**MAE validation của các mô hình khi tăng dữ liệu từ ít nhất lên toàn bộ:**

{md_table(gain)}

**Kết luận:** {c['crossover_text']}
"""


def section_conclusion(c):
    cmp_ = c["cmp"].set_index("model")
    return f"""## 6. Kết luận và hàm ý

- **Boosting:** sau tinh chỉnh trên toàn train, MAE CV là {cmp_.loc['hgb_best', 'mae_mean']:,.0f} EUR (mặc định {cmp_.loc['hgb_default', 'mae_mean']:,.0f}); Random Forest mặc định là {cmp_.loc['rf_default', 'mae_mean']:,.0f}. {c['hgb_vs_rf']}
- **Overfit:** {c['diag_short']}
- **Biến quyết định giá:** {c['top_text']}
- **Giả thuyết về Random Forest:** {c['crossover_short']}
- **Hàm ý cho báo cáo chính:** các kết quả tập test của tuần 3 **không đổi**. Nếu muốn so sánh boosting với bốn mô hình đã đóng băng trên test thì cần một tập giữ lại mới (dữ liệu mới hoặc chia lại từ đầu), không được dùng lại tập test hiện tại.

## 7. Giới hạn

- Boosting chưa được đánh giá trên test (chủ ý, xem mục 1); so sánh với các mô hình khác chỉ là CV trên train và bất đối xứng về mức độ tinh chỉnh.
- Độ quan trọng hoán vị tính trên tập con 20k và 5 fold; các cột tương quan chia sẻ độ quan trọng.
- Chẩn đoán overfit và đường cong học dựa trên MAE; không có kiểm định thống kê chính thức (5 fold).
- Chọn tốt nhất trong {N_HGB} cấu hình boosting làm MAE CV hơi lạc quan.
"""


def section_claims(claims):
    rows = pd.DataFrame([{"mã": i, "phát biểu": t, "kiểm tra với dữ liệu": "ĐÚNG" if ok else "KHÔNG ĐÚNG"} for i, t, ok in claims])
    return f"""## Phụ lục A. Tự kiểm chứng các phát biểu

Các phát biểu được máy tính kiểm tra lại với bảng số liệu mỗi lần sinh báo cáo. "KHÔNG ĐÚNG" là kết quả thật của thí nghiệm, không phải lỗi, và được nêu trong báo cáo.

{md_table(rows)}
"""


# ------------------------------------------------------------------------------------------ main
def build_context() -> dict:
    stats = json.loads((R / "stats.json").read_text(encoding="utf-8"))
    best = json.loads(HGB_BEST.read_text(encoding="utf-8"))
    cmp_ = _t("week3_hgb_vs_models_fullcv")
    diag, imp, cross, curves = _t("week3_diag_summary"), _t("week3_importance_summary"), _t("week3_rf_crossover"), _t("week3_learning_curves_summary")
    c = dict(best=best, cmp=cmp_, hgb_agg=_t("week3_hgb_summary"), diag=diag, imp=imp, cross=cross, curves=curves, banner="", n_train=int(stats["split"]["n_train"]))

    cm = cmp_.set_index("model")
    diff = cm.loc["hgb_best", "mae_mean"] - cm.loc["rf_default", "mae_mean"]
    wins = int(cm.loc["rf_default", "hgb_best_wins_folds"])
    c["hgb_vs_rf"] = (f"Boosting {'tốt hơn' if diff < 0 else 'kém hơn'} Random Forest mặc định {abs(diff):,.0f} EUR ({abs(diff) / cm.loc['rf_default', 'mae_mean'] * 100:.1f}%), "
                      f"thắng {wins}/5 fold; boosting huấn luyện nhanh hơn rất nhiều ({cm.loc['hgb_best', 'fit_seconds']:,.0f} s so với {cm.loc['rf_default', 'fit_seconds']:,.0f} s mỗi fold).")

    cr = cross.sort_values("fraction")
    small, full = cr.iloc[0], cr.iloc[-1]
    cross_ok = small["tuned_minus_default"] < 0 < full["tuned_minus_default"]
    c["crossover_short"] = ("được ủng hộ: tinh chỉnh tốt hơn ở dữ liệu nhỏ và tệ hơn ở toàn bộ dữ liệu." if cross_ok else
                            "chưa được ủng hộ đầy đủ bởi dữ liệu (xem mục 5).")
    def _who(r):
        return (f"tinh chỉnh tốt hơn mặc định {abs(r['tuned_minus_default']):,.0f} EUR" if r["tuned_minus_default"] < 0
                else f"tinh chỉnh tệ hơn mặc định {r['tuned_minus_default']:,.0f} EUR")
    mid = "; ".join(f"{r['fraction'] * 100:.0f}% ({r['n_train_used']:,.0f} dòng): {_who(r)}" for _, r in cr.iloc[1:-1].iterrows())
    c["crossover_text"] = (f"ở {small['fraction'] * 100:.0f}% dữ liệu ({small['n_train_used']:,.0f} dòng) {_who(small)} ({int(small['folds_tuned_better'])}/{int(small['n_folds'])} fold); "
                           + (f"{mid}; " if mid else "")
                           + f"ở 100% ({full['n_train_used']:,.0f} dòng) {_who(full)} ({int(full['folds_tuned_better'])}/{int(full['n_folds'])} fold tinh chỉnh tốt hơn). "
                           + ("**Đây đúng là kiểu đảo thứ hạng mà giả thuyết dự đoán**, nên giải thích 'siêu tham số chọn trên dữ liệu nhỏ không còn tối ưu trên dữ liệu lớn' được ủng hộ bằng dữ liệu." if cross_ok
                              else "Không thấy đảo thứ hạng rõ ràng, nên giải thích này chỉ còn là một khả năng; nguyên nhân khác (nhiễu fold, nhãn lỗi) chưa loại trừ được."))

    dd = imp[(imp["data"] == "dev20k")]
    ident = dd[dd["feature"].str.startswith("identity")]
    age = dd[dd["feature"].str.startswith("age")]
    tops = dd[(dd["kind"] == "single")].sort_values("delta_mae", ascending=False).groupby("name").head(3).groupby("name")["feature"].apply(list)
    id_by = ident.set_index("name")["delta_mae"]
    top1 = dd[dd["kind"] == "single"].sort_values("delta_mae", ascending=False).groupby("name").head(1).set_index("name")
    single_beats = [label(n) for n in id_by.index if top1.loc[n, "delta_mae"] > id_by[n]]
    c["top_text"] = (f"nhóm **định danh xe (hãng + mẫu)** làm MAE tăng trung bình {ident['delta_mae'].mean():,.0f} EUR khi bị xáo trộn, nhóm **tuổi xe** {age['delta_mae'].mean():,.0f} EUR; "
                     "ba cột đơn lẻ quan trọng nhất theo từng mô hình: " + "; ".join(f"{label(n)}: {', '.join(v)}" for n, v in tops.items()) + ". "
                     + (f"Ở {', '.join(single_beats)} một cột đơn lẻ (thường là `power_kw`) còn làm MAE tăng nhiều hơn cả nhóm định danh: công suất là chỉ báo thay thế cho "
                        "mẫu xe/phân khúc, nên mô hình dựa vào nó nhiều (giải thích hợp lý, chưa kiểm chứng riêng)." if single_beats else ""))
    dv = diag[diag["data"] == "dev20k"].set_index("name")
    rr = dv.drop(index="knn_default", errors="ignore")["valid_over_train_mae"]
    trees = rr[[n for n in rr.index if n.startswith(("rf", "hgb"))]]
    lo = rr.idxmin()
    c["diag_short"] = (f"khoảng cách validation/train lớn nhất ở nhóm cây ({trees.min():.2f}–{trees.max():.2f}) nhưng nhóm này vẫn có MAE validation thấp nhất; "
                       f"khoảng cách nhỏ nhất là {label(lo)} ({rr.min():.2f}) vì nó khớp train kém (underfitting), không phải vì nó tổng quát tốt hơn.")
    c["diag_text"] = (f"khoảng cách giữa MAE train và validation cho biết mô hình ghi nhớ dữ liệu huấn luyện đến đâu, nhưng **không** nên dùng riêng nó để loại mô hình. Các mô hình cây (Random Forest, boosting) có tỷ lệ "
                      f"{trees.min():.2f}–{trees.max():.2f} (MAE train thấp hơn validation rõ rệt) song vẫn có MAE validation thấp nhất, còn {label(lo)} có tỷ lệ nhỏ nhất ({rr.min():.2f}) nhưng "
                      f"MAE train ({dv.loc[lo, 'mae_train']:,.0f}) và R² train ({dv.loc[lo, 'r2_train']:.2f}) cho thấy nó khớp train kém (underfitting). KNN mặc định (trọng số khoảng cách) có MAE train gần 0 "
                      "vì mỗi điểm là láng giềng gần nhất của chính nó, nên tỷ lệ của nó vô nghĩa và không được vẽ. Điều quan trọng cho tổng quát hóa là **MAE validation**.")
    claims = verify_claims(c)
    c["claims"] = claims
    if any(not ok for _, _, ok in claims):
        c["banner"] = "\n> **Lưu ý:** có phát biểu ở Phụ lục A không đúng với dữ liệu; đó là kết quả thật và được nêu trong báo cáo.\n"
    return c


def write():
    c = build_context()
    failed = [(i, t) for i, t, ok in c["claims"] if not ok]
    for i, t in failed:
        print(f"[CLAIM NOT TRUE] {i}: {t}")
    body = "\n".join([section_intro(c), section_hgb(c), section_diag(c), section_importance(c), section_curves(c), section_conclusion(c),
                      section_claims(c["claims"])])
    (R / "week3_extra_report.md").write_text(body, encoding="utf-8")
    print("wrote", R / "week3_extra_report.md", "| claims:", len(c["claims"]), "| not true:", len(failed))
    return failed


if __name__ == "__main__":
    write()
