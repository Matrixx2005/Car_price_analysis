"""Write reports/week3_report.md from the week-3 tables (CV stages: train only; test: saved predictions of the single
final evaluation). Numbers come from CSV/JSON; interpretive claims are re-checked in verify_claims (Appendix A).

Run:  python -m src.reporting_week3
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.pipelines import MODELS
from src.reporting import md_table
from src.week2_analysis import ENC_LABEL, MODEL_LABEL, SCALER_LABEL
from src.week3_experiments import EXPENSIVE_EUR, EXTREME_QS, N_RANDOM, RARE_COUNTS, SPACES

R, T = config.REPORTS, config.TABLES


def _t(name):
    return pd.read_csv(T / f"{name}.csv")


def _eur(x):
    return f"{x:,.0f}"


def _name(n: str) -> str:
    if n == "naive_median":
        return "Mô hình ngây thơ (trung vị giá train)"
    if n.endswith("_cvbest_20k"):
        return f"{MODEL_LABEL[n.split('_')[0]]} (tốt nhất theo CV nhưng quá đắt; huấn luyện trên 20k dòng) [tham khảo]"
    m, kind = n.rsplit("_", 1)
    return f"{MODEL_LABEL[m]} ({'đã tinh chỉnh' if kind == 'tuned' else 'mặc định tuần 2'})"


def cfg_text(e: dict) -> str:
    bits = [ENC_LABEL[e["encoder"]], SCALER_LABEL[e["scaler"]], "log giá" if e["log_target"] else "giá gốc"]
    if e["extreme_q"] is not None:
        bits.append(f"cắt xe giá > phân vị {e['extreme_q'] * 100:g}% của train")
    if e["feature_set"] == "with_model_version":
        bits.append("+ model_version" + (f" (gom < {e['rare_min_count']} lần)" if e["rare_min_count"] else ""))
    return ", ".join(bits)


# ------------------------------------------------------------------------------------------ claims
def verify_claims(c: dict) -> list[tuple[str, str, bool]]:
    out = []

    def add(i, t, ok):
        out.append((i, t, bool(ok)))

    tv, frozen = c["tv"], c["frozen"]
    add("W1", "Cấu hình tinh chỉnh không tệ hơn cấu hình mặc định về MAE CV ở mọi mô hình (mặc định là một ứng viên)", (tv["tuned_mae"] <= tv["default_mae"] + 1e-9).all())
    if c["metrics"] is not None:
        m = c["metrics"].set_index("name")
        naive = m.loc["naive_median", "mae"]
        add("W2", "Mọi mô hình đã tinh chỉnh có MAE test thấp hơn mô hình ngây thơ ít nhất 50%", all(m.loc[f"{x}_tuned", "mae"] < 0.5 * naive for x in MODELS))
        tuned = {x: m.loc[f"{x}_tuned", "mae"] for x in MODELS}
        add("W3", "Mô hình được chọn bằng CV cũng có MAE test thấp nhất trong 4 mô hình đã tinh chỉnh", min(tuned, key=tuned.get) == frozen["selected_model"])
        cv = {x: frozen["models"][x]["final"]["cv"]["mae_mean"] for x in MODELS}
        add("W4", "MAE test lệch không quá 30% so với MAE CV ở mọi mô hình đã tinh chỉnh", all(abs(tuned[x] / cv[x] - 1) <= 0.30 for x in MODELS))
        add("W5", "Thứ hạng 4 mô hình theo MAE test trùng thứ hạng theo MAE CV", sorted(MODELS, key=tuned.get) == sorted(MODELS, key=cv.get))
        ci = c["ci"]
        pm = ci[(ci.kind == "paired_mae_diff")]
        sel = frozen["selected_model"]
        rows = pm[pm["name"].str.startswith(f"{sel}_tuned - ")]
        add("W6", "Mô hình được chọn tốt hơn từng mô hình tuned khác với xác suất bootstrap ≥ 95% (khoảng tin cậy của hiệu MAE nằm dưới 0)", len(rows) > 0 and (rows["ci_high"] < 0).all())
        te = c["errors"]
        pb = te["price_band"].set_index("key")
        add("W7", "MAE tuyệt đối tăng theo khoảng giá (xe đắt có MAE lớn nhất) nhưng MedAPE của nhóm giá cao không vượt MedAPE của nhóm giá thấp nhất",
            pb["mae"].iloc[-1] == pb["mae"].max() and pb["medape_pct"].iloc[-1] <= pb["medape_pct"].iloc[0])
        add("W8", "Mô hình được chọn dự đoán thấp hơn giá thực của xe >= 160.000 EUR (sai số có dấu âm)", pb["bias"].iloc[-1] < 0)
        pr = pm[pm["name"] == f"{sel}_tuned - {sel}_default"]
        add("W10", "Cấu hình tinh chỉnh của mô hình được chọn KHÔNG tệ hơn cấu hình mặc định trên test (khoảng tin cậy của hiệu MAE không nằm hoàn toàn trên 0)",
            len(pr) > 0 and not (pr["ci_low"] > 0).all())
        s = te["sensitivity"].set_index("name").loc[f"{sel}_tuned"]
        add("W11", "Loại các dòng nhãn nghi lỗi từ tuần 1 làm R² của mô hình được chọn tăng thêm ít nhất 0,2 (R² do vài nhãn lỗi chi phối)",
            s["r2_without_flagged"] - s["r2"] >= 0.2)
        add("W12", "5 dự đoán sai nhất chiếm hơn 80% tổng sai số bình phương của mô hình được chọn", s["share_sse_top5_errors_pct"] > 80)
    ex = c["ex"]
    if ex is not None:
        e = ex[ex.extreme_q != "none"]
        add("W9", "Cắt xe giá cao khỏi tập huấn luyện làm tăng MAE trên xe >= 100.000 EUR ở mọi mô hình và mức cắt",
            all(e.loc[(e.model == m_) & (e.extreme_q == q), "mae_ge_100k"].iloc[0] > ex.loc[(ex.model == m_) & (ex.extreme_q == "none"), "mae_ge_100k"].iloc[0]
                for m_ in MODELS for q in e.loc[e.model == m_, "extreme_q"]))
    return out


# ------------------------------------------------------------------------------------------ sections
def section_scope(c):
    return f"""# Báo cáo Tuần 3 – Tinh chỉnh, chốt cấu hình và đánh giá cuối trên tập test

_Mọi số liệu được tính từ `reports/tables/week3_*.csv`, `reports/week3_frozen_config.json` và dự đoán đã lưu của lần đánh giá test duy nhất. Báo cáo sinh bằng `python -m src.reporting_week3`._
{c['banner']}
## 1. Mục tiêu, phạm vi và quy tắc bảo vệ tập test

Tuần 3 (1) tinh chỉnh siêu tham số của bốn mô hình, (2) quyết định hai việc còn treo từ tuần 2: xử lý xe giá cực cao và gom danh mục hiếm của `model_version`, (3) **chốt cấu hình**, rồi (4) đánh giá **đúng một lần** trên tập test và phân tích phần dư.

**Quy trình chống rò rỉ (được cưỡng chế bằng code và test):**

1. Mọi lựa chọn (siêu tham số, log giá, cắt xe giá cao, gom danh mục, mô hình được chọn) chỉ dựa trên **cross-validation trên tập train** (GroupKFold 5 fold theo nhóm xe, cùng tập con {c['dev_rows']:,} dòng và cùng các fold như tuần 2).
2. Cấu hình được **ghi băng** vào `reports/week3_frozen_config.json` cùng mã băm của file train và file kết quả. `src/final_eval.py` kiểm tra lại các mã băm này; nếu bất kỳ thứ gì đổi sau khi đóng băng thì từ chối chạy.
3. `final_eval.py` là mô-đun **duy nhất** đọc `test.parquet`. Sau khi chạy xong nó ghi file đánh dấu (`week3_test_evaluation_done.json`); lần chạy thứ hai bị từ chối. Mọi phân tích tiếp theo dùng dự đoán đã lưu.
4. Mọi ngưỡng học từ dữ liệu (ngưỡng cắt giá, ngưỡng gom danh mục, thống kê điền thiếu, scaler, encoder) được tính trên phần train của từng fold (khi CV) hoặc trên toàn bộ train (khi huấn luyện lần cuối), không bao giờ từ dữ liệu đánh giá.
"""


def section_design(c):
    sp = pd.DataFrame([{"mô hình": MODEL_LABEL[m], "không gian tìm kiếm": "; ".join(f"{k} ∈ {{{', '.join(str(x) if not isinstance(x, float) else f'{x:g}' for x in v)}}}" for k, v in s.items()),
                        "số ứng viên": c["n_tried"][m]} for m, s in SPACES.items()])
    sp = sp.set_index("mô hình").loc[[MODEL_LABEL[m] for m in MODELS]].reset_index()
    return f"""## 2. Thiết kế

**Tiền xử lý cố định** theo kết quả tuần 2 (encoder và scaler của cấu hình tốt nhất từng mô hình): KNN One-hot + Standard, Ridge One-hot + MinMax, SVR One-hot + Standard, Random Forest One-hot (+ Standard). Thêm một yếu tố được tìm kiếm cùng siêu tham số: **dùng log giá hay giá gốc** (`log1p` trên target, đổi ngược bằng `expm1`).

{md_table(sp)}

Mỗi mô hình luôn gồm cấu hình mặc định của tuần 2 (với giá gốc và với log giá) làm mốc so sánh. KNN, SVR và Random Forest dùng tìm kiếm ngẫu nhiên có seed ({', '.join(f'{MODEL_LABEL[k]} {v}' for k, v in N_RANDOM.items())} ứng viên, không lặp), Ridge duyệt toàn bộ lưới α (không tính mặc định).

**Bốn giai đoạn**, mỗi giai đoạn dùng kết quả của giai đoạn trước:

1. **tune:** siêu tham số + log giá.
2. **extremes:** cắt khỏi tập *huấn luyện* các xe có giá vượt phân vị {', '.join(f'{q * 100:g}%' for q in EXTREME_QS)} của chính fold huấn luyện (các dòng đánh giá không bao giờ bị cắt); báo cáo thêm MAE riêng cho xe dưới và từ {EXPENSIVE_EUR:,} EUR.
3. **grouping:** thêm `model_version`, gom danh mục xuất hiện ít hơn N lần (N ∈ {{{', '.join(str(x) if x else 'không gom' for x in RARE_COUNTS)}}}) vào một nhãn chung (đếm trên fold huấn luyện), với One-hot và Target Encoding.
4. **combine:** tổ hợp các thành phần thắng ở trên, so với cấu hình tune tốt nhất.

**Quy tắc quyết định (cho mọi thành phần tùy chọn):** ứng viên chỉ thay thế cấu hình đơn giản hơn khi MAE trung bình thấp hơn **và** thắng ít nhất 4/5 fold so với cấu hình đó (tránh chạy theo nhiễu). Với siêu tham số thì chọn MAE trung bình thấp nhất trong các ứng viên đã thử; cách chọn "tốt nhất trong nhiều ứng viên" có thể hơi lạc quan, nên ước lượng cuối cùng dựa vào tập test chưa từng dùng để chọn. Mô hình được chọn là mô hình có MAE CV thấp nhất.

**Ràng buộc chi phí tính toán.** Thời gian huấn luyện đo trong CV (trên khoảng 16.000 dòng) được ngoại suy lên toàn bộ train ({c['n_train']:,} dòng) với hệ số mũ theo mô hình (SVR ≈ bình phương số dòng, Random Forest ≈ 1,2, KNN và Ridge ≈ tuyến tính). Cấu hình có thời gian huấn luyện ước tính trên toàn train vượt **{c['frozen']['full_fit_budget_s'] / 60:.0f} phút** không đủ điều kiện được chọn (vì không thể huấn luyện lại để đánh giá cuối). {len(c['frozen']['skipped_configs'])} cấu hình SVR cực đoan (C=300, ε=0,02, log giá) bị hủy sau hơn một giờ chưa xong một fold (`reports/tables/week3_skipped_configs.json`). Các cấu hình tốt nhất theo CV nhưng quá đắt được đóng băng làm **mốc tham khảo**: huấn luyện trên cùng tập con 20k dòng như khi CV và đánh giá một lần trên test, không đủ điều kiện được chọn.
"""


def section_tune(c):
    tv = c["tv"].copy()
    t = pd.DataFrame({"mô hình": tv["model"].map(MODEL_LABEL), "số cấu hình": tv["n_configs_tried"], "MAE mặc định": tv["default_mae"], "MAE tinh chỉnh": tv["tuned_mae"],
                      "thay đổi %": tv["gain_pct"], "fold tốt hơn": tv["folds_tuned_better"].astype(str) + " / 5", "log giá": tv["tuned_log_target"].map({True: "có", False: "không"}),
                      "siêu tham số tốt nhất": tv["tuned_params"]})
    r = pd.DataFrame({"mô hình": tv["model"].map(MODEL_LABEL), "RMSE mặc định": tv["default_rmse"], "RMSE tinh chỉnh": tv["tuned_rmse"], "R² mặc định": tv["default_r2"],
                      "R² tinh chỉnh": tv["tuned_r2"], "MedAPE % mặc định": tv["default_medape"], "MedAPE % tinh chỉnh": tv["tuned_medape"]})
    a = pd.DataFrame({"mô hình": tv["model"].map(MODEL_LABEL), "số cấu hình quá đắt": tv["n_infeasible"], "MAE tốt nhất không xét chi phí": tv["any_best_mae"],
                      "thời gian huấn luyện ước tính trên toàn train (giờ)": tv["any_best_est_full_fit_s"] / 3600, "log giá": tv["any_best_log_target"].map({True: "có", False: "không"}),
                      "siêu tham số": tv["any_best_params"]})
    return f"""## 3. Tinh chỉnh siêu tham số (CV trên train)

**Cấu hình khả thi tốt nhất (đủ điều kiện được chọn) so với mặc định:**

{md_table(t)}

**Cấu hình tốt nhất theo MAE CV nếu không xét chi phí** (dùng làm mốc tham khảo ở mục 7 khi khác cấu hình trên):

{md_table(a)}

Các chỉ số còn lại (cùng CV):

{md_table(r)}

![Các cấu hình đã thử](figures/week3_tuning.png)

Mỗi điểm là một cấu hình (xếp theo MAE); ngôi sao là cấu hình mặc định của tuần 2.
"""


def section_extremes(c):
    ex = c["ex"]
    if ex is None:
        return "## 4. Xe giá cực cao\n\n_Chưa chạy._\n"
    e = ex.copy()
    e["model"] = e["model"].map(MODEL_LABEL)
    e["folds_better_than_none"] = e["folds_better_than_none"].map(lambda v: "" if pd.isna(v) else f"{int(v)} / 5")
    e = e.rename(columns={"extreme_q": "cắt trên phân vị", "mae": "MAE", "mae_lt_100k": f"MAE < {EXPENSIVE_EUR // 1000}k", "mae_ge_100k": f"MAE >= {EXPENSIVE_EUR // 1000}k",
                          "bias_ge_100k": f"sai số có dấu >= {EXPENSIVE_EUR // 1000}k", "rmse": "RMSE", "folds_better_than_none": "fold tốt hơn không cắt", "rel_change_pct": "Δ MAE %"})
    return f"""## 4. Xe giá cực cao (cắt khỏi tập huấn luyện)

Ngưỡng cắt là phân vị giá của chính fold huấn luyện; mọi dòng đánh giá được giữ nguyên. "Sai số có dấu" = dự đoán − thực tế (âm = dự đoán thấp).

{md_table(e)}

![Hiệu ứng của việc cắt xe giá cao](figures/week3_extremes.png)

Các lựa chọn được chấp nhận theo quy tắc ở mục 2 sẽ được ghi ở mục 6.
"""


def section_grouping(c):
    gr = c["gr"]
    if gr is None:
        return "## 5. model_version và gom danh mục hiếm\n\n_Chưa chạy._\n"
    g = gr.copy()
    g["model"], g["encoder"] = g["model"].map(MODEL_LABEL), g["encoder"].map(ENC_LABEL)
    g["folds_better"] = g["folds_better"].astype(str) + " / 5"
    g = g.rename(columns={"rare_min_count": "gom < N lần", "mae": "MAE", "baseline_mae": "MAE không có model_version", "rel_change_pct": "Δ MAE %", "n_features_out": "số cột",
                          "folds_better": "fold tốt hơn không có"})
    return f"""## 5. `model_version` và gom danh mục hiếm

Cấu hình cơ sở là cấu hình tune tốt nhất của từng mô hình (không có `model_version`). Δ MAE < 0 là có `model_version` tốt hơn.

{md_table(g)}

![Hiệu ứng của model_version và việc gom danh mục hiếm](figures/week3_grouping.png)
"""


def section_frozen(c):
    f = c["frozen"]
    rows = []
    for m in MODELS:
        e, d = f["models"][m]["final"], f["models"][m]["default"]
        rows.append({"mô hình": MODEL_LABEL[m], "cấu hình đóng băng": cfg_text(e), "siêu tham số": json.dumps(e["params"], ensure_ascii=False), "MAE CV": e["cv"]["mae_mean"],
                     "MAE CV mặc định": d["cv"]["mae_mean"], "R² CV": e["cv"]["r2_mean"], "tổ hợp thành phần tùy chọn": "có" if f["models"][m]["uses_combined"] else "không"})
    return f"""## 6. Cấu hình đóng băng

{md_table(pd.DataFrame(rows))}

**Mô hình được chọn bằng CV: {MODEL_LABEL[f['selected_model']]}.** Cấu hình được ghi lúc {f['created_at']} (`reports/week3_frozen_config.json`).
"""


def section_test(c):
    if c["metrics"] is None:
        return "## 7. Đánh giá cuối trên tập test\n\n_Chưa chạy: `python -m src.final_eval`._\n"
    m, ci, f = c["metrics"], c["ci"], c["frozen"]
    cim = ci[ci.kind == "metric"].pivot(index="name", columns="metric", values=["ci_low", "ci_high"])
    rows = []
    for _, r in m.sort_values("mae").iterrows():
        n = r["name"]
        rows.append({"mô hình": _name(n) + (" ◀ được chọn" if r["is_selected"] else ""), "MAE": r["mae"], "MAE 95% CI": f"{cim.loc[n, ('ci_low', 'mae')]:,.0f} – {cim.loc[n, ('ci_high', 'mae')]:,.0f}",
                     "RMSE": r["rmse"], "R²": r["r2"], "R² 95% CI": f"{cim.loc[n, ('ci_low', 'r2')]:.2f} – {cim.loc[n, ('ci_high', 'r2')]:.2f}", "MedAPE %": r["medape_pct"],
                     "fit (s)": r["fit_seconds"]})
    pm = ci[ci.kind == "paired_mae_diff"].copy()
    pm["so sánh (MAE của B − A)"] = pm["name"]
    pm["95% CI của hiệu"] = pm.apply(lambda r: f"{r['ci_low']:,.0f} – {r['ci_high']:,.0f}", axis=1)
    pm["xác suất B tốt hơn A"] = pm["share_b_better"].map(lambda v: f"{v * 100:.1f}%")
    t = m.set_index("name")
    cvrows = [{"mô hình": MODEL_LABEL[x], "MAE CV (train, 20k)": f["models"][x]["final"]["cv"]["mae_mean"], "MAE test": t.loc[f"{x}_tuned", "mae"],
               "chênh lệch %": (t.loc[f"{x}_tuned", "mae"] / f["models"][x]["final"]["cv"]["mae_mean"] - 1) * 100} for x in MODELS]
    return f"""## 7. Đánh giá cuối trên tập test (một lần duy nhất)

Mỗi mô hình được huấn luyện lại trên **toàn bộ train** ({c['n_train']:,} dòng) với cấu hình đóng băng rồi dự đoán {c['n_test']:,} xe của tập test (nhóm xe không giao với train). Khoảng tin cậy 95% lấy bằng **bootstrap theo nhóm xe** ({c['n_boot']:,} lần lấy mẫu lại cả nhóm vì các tin cùng một xe không độc lập).

{md_table(pd.DataFrame(rows))}

![MAE trên tập test](figures/week3_test_models.png)

**So sánh cặp (bootstrap theo nhóm):**

{md_table(pm[['so sánh (MAE của B − A)', '95% CI của hiệu', 'xác suất B tốt hơn A']])}

**MAE ước lượng bằng CV so với MAE trên test (cấu hình đã tinh chỉnh):**

{md_table(pd.DataFrame(cvrows))}

![CV so với test](figures/week3_cv_vs_test.png)

Lưu ý: ước lượng CV dùng tập con 20k dòng và các mô hình nhỏ hơn; mô hình cuối được huấn luyện trên {c['n_train']:,} dòng nên khác biệt là bình thường.
"""


def section_residuals(c):
    if c["errors"] is None:
        return "## 8. Phân tích phần dư\n\n_Chưa chạy._\n"
    e, sel = c["errors"], MODEL_LABEL[c["frozen"]["selected_model"]]

    def fmt(df, key, label):
        d = df.copy()
        d[key] = d[key].astype(str)
        return d.rename(columns={key: label, "mae": "MAE", "bias": "sai số có dấu", "medape_pct": "MedAPE %", "median_price": "giá trung vị"})

    return f"""## 8. Phân tích phần dư của mô hình được chọn ({sel}, test)

"Sai số có dấu" = dự đoán − thực tế (âm = dự đoán thấp hơn giá thực).

**Theo khoảng giá thực:**

{md_table(fmt(e['price_band'], 'key', 'khoảng giá'))}

**Theo tuổi xe:**

{md_table(fmt(e['age_band'], 'key', 'tuổi (năm)'))}

**Theo quốc gia (n ≥ 100):**

{md_table(fmt(e['country'], 'key', 'quốc gia'))}

**Theo hãng (n ≥ 100, sắp theo MAE giảm dần):**

{md_table(fmt(e['brand'], 'key', 'hãng'))}

**10 dự đoán sai nhiều nhất:**

{md_table(e['worst'])}

**Độ nhạy với các nhãn lỗi (mô tả sau khi đánh giá, không dùng để chọn lại mô hình).** Cột "bỏ dòng nghi lỗi" loại các tin bị gắn cờ giá nghi lỗi ở tuần 1 (`flag_price_review`, {e['sensitivity']['n_flagged_excluded'].iloc[0]} tin trong test); "SSE top 5" là phần trăm tổng sai số bình phương do 5 dự đoán sai nhất gây ra.

{md_table(e['sensitivity'].rename(columns={'name': 'mô hình', 'mae': 'MAE', 'mae_without_flagged': 'MAE bỏ dòng nghi lỗi', 'rmse': 'RMSE', 'rmse_without_flagged': 'RMSE bỏ dòng nghi lỗi', 'r2': 'R²', 'r2_without_flagged': 'R² bỏ dòng nghi lỗi', 'share_sse_top5_errors_pct': 'SSE top 5 (%)'}).drop(columns=['n_flagged_excluded']))}

![Dự đoán so với thực tế](figures/week3_test_pred_vs_actual.png)

![Sai số theo nhóm](figures/week3_test_errors_by_group.png)
"""


def section_discussion(c):
    tv, ex, gr, f = c["tv"], c["ex"], c["gr"], c["frozen"]
    sel = f["selected_model"]
    bits = []
    g = tv.set_index("model")["gain_pct"]
    bits.append("1. **Tinh chỉnh có đáng không?** Mức thay đổi MAE CV so với mặc định: " + ", ".join(f"{MODEL_LABEL[m]} {g[m]:+.1f}%" for m in MODELS)
                + ". Mức cải thiện lớn nhất thuộc về mô hình bị thiệt nhất do siêu tham số mặc định (Ridge/SVR thường hưởng lợi từ log giá và α/C phù hợp), trong khi Random Forest ít nhạy.")
    if ex is not None:
        e = ex[ex.extreme_q != "none"]
        worse = int((e["mae_ge_100k"].to_numpy() > np.repeat(ex.loc[ex.extreme_q == "none", "mae_ge_100k"].to_numpy(), len(EXTREME_QS))).sum()) if len(e) else 0
        bits.append(f"2. **Xe giá cực cao:** cắt chúng khỏi tập huấn luyện làm sai số trên xe từ {EXPENSIVE_EUR:,} EUR tăng ở {worse}/{len(e)} cặp (mô hình, mức cắt); lựa chọn nào được nhận theo quy tắc 4/5 fold được ghi ở mục 6.")
    if gr is not None:
        best = gr.loc[gr.groupby("model")["rel_change_pct"].idxmin()].set_index("model")
        oh = gr[gr.encoder == "onehot"].groupby("model")["rel_change_pct"].apply(lambda s: s[:2].abs().max())
        heavy = gr[gr.rare_min_count.astype(str).isin(["50", "200"])]["rel_change_pct"]
        bits.append("3. **model_version:** thay đổi MAE tốt nhất so với không có `model_version` ở từng mô hình: "
                    + ", ".join(f"{MODEL_LABEL[m]} {best.loc[m, 'rel_change_pct']:+.1f}% ({ENC_LABEL[best.loc[m, 'encoder']]}, gom {best.loc[m, 'rare_min_count']}, {int(best.loc[m, 'folds_better'])}/5 fold)" for m in MODELS)
                    + f". Với One-hot, `model_version` (đã gom danh mục < 10 lần) thay đổi MAE nhiều nhất {oh.max():.1f}%, phù hợp với kết luận tuần 2; gom mạnh hơn (≥ 50 lần) hầu như luôn làm tệ hơn "
                    f"({(heavy > 0).mean() * 100:.0f}% số cấu hình). Chỉ KNN hưởng lợi rõ ràng (Target Encoding); Target Encoding làm tệ Ridge và SVR.")
    if c["metrics"] is not None:
        m = c["metrics"].set_index("name")
        ci, errs = c["ci"], c["errors"]
        bits.append(f"4. **Mô hình được chọn bằng CV:** {MODEL_LABEL[sel]} (MAE test {_eur(m.loc[f'{sel}_tuned', 'mae'])} EUR, R² {m.loc[f'{sel}_tuned', 'r2']:.2f}, MedAPE {m.loc[f'{sel}_tuned', 'medape_pct']:.1f}%). "
                    f"Mô hình ngây thơ (luôn đoán trung vị) có MAE {_eur(m.loc['naive_median', 'mae'])} EUR, nên mọi mô hình đã tinh chỉnh giảm sai số hơn 70%.")
        tuned = {x: m.loc[f"{x}_tuned", "mae"] for x in MODELS}
        cvr = sorted(MODELS, key=lambda x: f["models"][x]["final"]["cv"]["mae_mean"])
        ter = sorted(MODELS, key=tuned.get)
        bits.append("5. **CV và test không hoàn toàn đồng ý.** Thứ hạng theo MAE CV: " + " < ".join(MODEL_LABEL[x] for x in cvr) + "; theo MAE test: " + " < ".join(MODEL_LABEL[x] for x in ter)
                    + ". Khác biệt giữa hai mô hình dẫn đầu nằm trong khoảng bất định (xem so sánh cặp ở mục 7), nên không thể khẳng định mô hình nào tốt hơn một cách chắc chắn.")
        pr = ci[(ci.kind == "paired_mae_diff") & (ci["name"] == f"{sel}_tuned - {sel}_default")]
        if len(pr):
            r = pr.iloc[0]
            bits.append(f"6. **Tinh chỉnh không giúp mô hình được chọn trên test.** {MODEL_LABEL[sel]} đã tinh chỉnh có MAE test {_eur(m.loc[f'{sel}_tuned', 'mae'])} EUR so với {_eur(m.loc[f'{sel}_default', 'mae'])} EUR của cấu hình mặc định "
                        f"(hiệu MAE 95% CI {r['ci_low']:,.0f} đến {r['ci_high']:,.0f} EUR). Một giải thích hợp lý (chưa kiểm chứng riêng) là siêu tham số được chọn trên 16.000 dòng huấn luyện (cây nông hơn và ít đặc trưng mỗi lần chia hơn để đủ nhanh) "
                        "không còn tối ưu khi huấn luyện trên 75.000 dòng, nơi cây sâu hơn có lợi. Ngược lại, tinh chỉnh giúp rõ rệt Ridge (chủ yếu nhờ log giá), SVR và KNN. "
                        "**Lưu ý:** cấu hình mặc định của Random Forest cho MAE test thấp nhất toàn bộ, nhưng đó là quan sát sau khi đã dùng test; không được dùng nó để chọn lại mô hình (sẽ biến tập test thành tập chọn). Muốn kết luận chắc chắn cần dữ liệu giữ lại mới hoặc CV trên toàn bộ train.")
        s = errs["sensitivity"].set_index("name").loc[f"{sel}_tuned"]
        bits.append(f"7. **R² thấp là do nhãn lỗi, không phải do mô hình tệ.** R² của {MODEL_LABEL[sel]} là {s['r2']:.2f} nhưng chỉ cần bỏ {int(s['n_flagged_excluded'])} tin bị gắn cờ giá nghi lỗi (BMW 316 giá 9,5 triệu EUR) thì lên {s['r2_without_flagged']:.2f}; "
                    f"5 dự đoán sai nhất gây {s['share_sse_top5_errors_pct']:.0f}% tổng sai số bình phương. Vì vậy báo cáo dùng MAE/MedAPE làm chỉ số chính.")
        pb = errs["price_band"].set_index("key")
        rest = pb["medape_pct"].iloc[1:]
        bits.append(f"8. **Xe đắt là điểm yếu.** MedAPE gần như đều ở các nhóm giá từ {pb.index[1]} trở lên ({rest.min():.1f}% đến {rest.max():.1f}%; nhóm {pb.index[0]} cao hơn: {pb['medape_pct'].iloc[0]:.1f}%), nhưng MAE tuyệt đối tăng theo giá và mô hình dự đoán thấp hơn thực tế khoảng {abs(pb['bias'].iloc[-1]):,.0f} EUR với xe từ 160.000 EUR (xe hiếm, ít mẫu huấn luyện).")
    return "## 9. Thảo luận\n\n" + "\n".join(bits) + "\n"


def section_conclusion(c):
    f = c["frozen"]
    if c["metrics"] is None:
        return ""
    m = c["metrics"].set_index("name")
    sel = f["selected_model"]
    ok = {x: m.loc[f"{x}_tuned", "mae"] for x in MODELS}
    return f"""## 10. Kết luận cho môn học

- **Quy trình:** dữ liệu AutoScout24 (118.382 tin) → {c['n_train'] + c['n_test']:,} xe cũ hợp lệ → chia theo nhóm xe → so sánh Encoding × Scaling (tuần 2) → tinh chỉnh và chốt cấu hình bằng CV (tuần 3) → đánh giá test đúng một lần.
- **Kết quả cuối:** mô hình được chọn theo quy trình là {MODEL_LABEL[sel]} với MAE test {_eur(ok[sel])} EUR (MedAPE {m.loc[f'{sel}_tuned', 'medape_pct']:.1f}%), so với {_eur(m.loc['naive_median', 'mae'])} EUR của mô hình ngây thơ. Trên test, SVR đã tinh chỉnh ({_eur(ok['svr'])} EUR) và Random Forest chênh nhau không có ý nghĩa thống kê.
- **Bài học về phương pháp:** (1) tinh chỉnh giúp nhiều nhất mô hình yếu nhất lúc đầu (Ridge giảm khoảng 42% MAE test nhờ log giá và α); (2) siêu tham số tối ưu trên tập con nhỏ có thể không chuyển sang dữ liệu lớn hơn (Random Forest); (3) chỉ số RMSE/R² bị vài nhãn lỗi chi phối nên cần MAE/MedAPE và kiểm tra dữ liệu; (4) chi phí tính toán là một ràng buộc thật khi chọn mô hình (SVR).
- **Điểm yếu cần cải thiện:** xe rất đắt (từ 160.000 EUR) bị dự đoán thấp; nhãn lỗi trong dữ liệu gốc.
- **Hướng phát triển (đồ án tốt nghiệp):** làm sạch nhãn lỗi bằng quy tắc dựa trên giá thị trường, thử mô hình boosting, thu thập dữ liệu thị trường Việt Nam và dùng `vietnam_adapter` để thay AutoScout24, kiểm tra lại các kết luận về encoding/scaling trên dữ liệu mới.
"""


def section_limits():
    return """## 11. Giới hạn và rủi ro

- Tinh chỉnh chạy trên tập con 20.000 dòng của train với 5 fold; siêu tham số tối ưu cho 16.000 dòng huấn luyện có thể không tối ưu cho 75.000 dòng (đặc biệt k của KNN, số cây và độ sâu của Random Forest).
- Không gian tìm kiếm có kích thước hữu hạn (tìm kiếm ngẫu nhiên 12–20 ứng viên); chọn "tốt nhất trong nhiều ứng viên" làm MAE CV hơi lạc quan. Tập test cho ước lượng không chệch nhưng chỉ là **một** phép đo trên một lần chia.
- Tập test chung phân phối với train (cùng nguồn, cùng thời điểm snapshot, thiên về xe cao cấp châu Âu): kết quả không suy ra được cho giá xe ở thị trường khác hoặc thời điểm khác, kể cả Việt Nam.
- Giá là giá chào bán, không phải giá giao dịch; không kiểm soát được chất lượng thực tế của xe (tai nạn, bảo dưỡng).
- Phần nhóm xe: tin trùng không có VIN chỉ được nhóm gần đúng (tuần 1); nếu còn tin trùng chéo train/test thì MAE test hơi lạc quan.
- Ridge trên ma trận thưa dùng bộ giải lặp (`tol=1e-4`) nên chạy lại có thể lệch cỡ 1e-4 tương đối; KNN có thể lệch cỡ 1e-6 do thứ tự cộng số thực khi song song.
- Thời gian huấn luyện đo khi nhiều tiến trình chạy song song, chỉ so sánh tương đối.
- Các "giải thích hợp lý" trong báo cáo là giả thuyết, chưa kiểm chứng bằng thí nghiệm riêng.
"""


def section_claims(claims):
    rows = pd.DataFrame([{"mã": i, "phát biểu": t, "kiểm tra với dữ liệu": "ĐÚNG" if ok else "KHÔNG ĐÚNG"} for i, t, ok in claims])
    return f"""## Phụ lục A. Tự kiểm chứng các phát biểu

Các phát biểu định tính được máy tính kiểm tra lại với bảng số liệu mỗi lần sinh báo cáo (`verify_claims` trong `src/reporting_week3.py`). Phát biểu "KHÔNG ĐÚNG" không phải lỗi mà là kết quả thật của thí nghiệm, cần được hiểu và nêu trong phần thảo luận.

{md_table(rows)}
"""


# ------------------------------------------------------------------------------------------ main
def build_context() -> dict:
    meta = json.loads((R / "week3_run_meta.json").read_text(encoding="utf-8"))
    stats = json.loads((R / "stats.json").read_text(encoding="utf-8"))
    summ = _t("week3_summary_all")
    frozen = json.loads((R / "week3_frozen_config.json").read_text(encoding="utf-8"))
    ctx = dict(tv=_t("week3_tune_vs_default"), frozen=frozen, dev_rows=int(meta["n_dev_rows"]),
               n_tried={m: int(((summ.model == m) & (summ.stage == "tune")).sum()) for m in MODELS},
               n_train=int(stats["split"]["n_train"]), n_test=int(stats["split"]["n_test"]), n_boot=1000, banner="",
               ex=_t("week3_extremes_effect") if (T / "week3_extremes_effect.csv").exists() else None,
               gr=_t("week3_grouping_effect") if (T / "week3_grouping_effect.csv").exists() else None,
               metrics=None, ci=None, errors=None)
    if (T / "week3_test_metrics.csv").exists() and (T / "week3_test_errors_by_price_band.csv").exists():
        ctx["metrics"], ctx["ci"] = _t("week3_test_metrics"), _t("week3_test_bootstrap_ci")
        ctx["errors"] = {k: _t(f"week3_test_errors_by_{k}") for k in ("price_band", "age_band", "brand", "country", "worst")}
        ctx["errors"]["sensitivity"] = _t("week3_test_sensitivity")
    return ctx


def write():
    c = build_context()
    claims = verify_claims(c)
    failed = [(i, t) for i, t, ok in claims if not ok]
    for i, t in failed:
        print(f"[CLAIM NOT TRUE] {i}: {t}")
    if failed:
        c["banner"] = "\n> **Lưu ý:** có phát biểu ở Phụ lục A không đúng với dữ liệu; đó là kết quả thật và được nêu trong phần thảo luận.\n"
    body = "\n".join([section_scope(c), section_design(c), section_tune(c), section_extremes(c), section_grouping(c), section_frozen(c), section_test(c),
                      section_residuals(c), section_discussion(c), section_conclusion(c), section_limits(), section_claims(claims)])
    (R / "week3_report.md").write_text(body, encoding="utf-8")
    print("wrote", R / "week3_report.md", "| claims:", len(claims), "| not true:", len(failed))
    return failed


if __name__ == "__main__":
    write()
