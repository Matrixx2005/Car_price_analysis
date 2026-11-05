"""Write reports/week2_report.md from the week-2 result tables.

Numbers are read from CSVs. The interpretive sentences make claims ("One-hot beats Ordinal in 0/5 folds"...);
every such claim is re-checked against the tables (verify_claims) and listed in Appendix A. If a claim is no
longer true, the report says so at the top and this script prints a warning.

Run:  python -m src.reporting_week2   (after src.experiments and src.week2_analysis)
"""
from __future__ import annotations

import json

import pandas as pd

from src import config
from src.reporting import md_table
from src.week2_analysis import ENC_LABEL, MODEL_LABEL, SCALER_LABEL

R, T = config.REPORTS, config.TABLES
MODELS = ["knn", "ridge", "svr", "rf"]
OTHER_ENCS = ["ordinal", "target", "frequency"]


def _t(name):
    return pd.read_csv(T / f"{name}.csv")


def _eur(x):
    return f"{x:,.0f}"


def _lab(df, cols=("model", "encoder", "scaler", "level")):
    df = df.copy()
    for c, m in (("model", MODEL_LABEL), ("encoder", ENC_LABEL), ("scaler", SCALER_LABEL), ("level", {**ENC_LABEL, **SCALER_LABEL})):
        if c in df and c in cols:
            df[c] = df[c].map(m)
    return df


# ------------------------------------------------------------------------------------------ claims
def verify_claims(c: dict) -> list[tuple[str, str, bool]]:
    """Each item: (id, claim written in the prose, is it true in the current tables?)."""
    pe, ps, lm, lr, lfm, mv, summ, conf = (c[k] for k in ("pe", "ps", "lm", "lr", "lfm", "mv", "summ", "conf"))
    three = ["knn", "rf", "svr"]
    out = []

    def add(cid, text, ok):
        out.append((cid, text, bool(ok)))

    add("C1", "Ordinal và Frequency không tốt hơn One-hot ở bất kỳ fold nào (KNN, Random Forest, SVR; scaler Standard)",
        (pe[pe.model.isin(three) & pe.level.isin(["ordinal", "frequency"])]["folds_better"] == 0).all())
    add("C2", "Target Encoding tốt hơn One-hot nhiều nhất 1/5 fold (KNN, Random Forest, SVR)",
        (pe[pe.model.isin(three) & (pe.level == "target")]["folds_better"] <= 1).all())
    r = pe[pe.model == "ridge"]
    add("C3", "Trên tập con, Ridge: chênh lệch giữa các encoder ≤ 5% và không encoder nào tốt hơn One-hot quá 2/5 fold",
        r["rel_change_pct"].abs().max() <= 5 and (r["folds_better"] <= 2).all())
    s = ps[(ps.level == "standard") & ps.model.isin(["knn", "svr"])]
    add("C4", "Scaler Standard tốt hơn No scaling ở 20/20 trường hợp của KNN và của SVR", (s["folds_better"].groupby(s["model"]).sum() == s["n"].groupby(s["model"]).sum()).all())
    add("C5", "Random Forest bất biến với scaler (thay đổi MAE < 0,1%)", ps[ps.model == "rf"]["rel_change_pct"].abs().max() < 0.1)
    add("C6", "Log giá làm giảm MAE của Ridge ở 5/5 fold với mọi encoder",
        ((lm[lm.model == "ridge"]["folds_b_better"] == 5) & (lm[lm.model == "ridge"]["rel_change_pct"] < 0)).all())
    add("C7", "Log giá làm RMSE tệ hơn ở KNN và Random Forest (trừ Target Encoding)",
        (lr[lr.model.isin(["knn", "rf"]) & (lr.encoder != "target")]["rel_change_pct"] > 0).all())
    o = mv[(mv.encoder == "ordinal") & mv.model.isin(three)]
    add("C8", "Thêm model_version làm Ordinal tệ hơn ở 0/5 fold tốt hơn (KNN, Random Forest, SVR)", ((o["folds_b_better"] == 0) & (o["rel_change_pct"] > 0)).all())
    add("C9", "Thêm model_version làm MAE của One-hot đổi ≤ 0,5% ở mọi mô hình", mv[mv.encoder == "onehot"]["rel_change_pct"].abs().max() <= 0.5)
    m = summ[(summ.experiment == "main") & (summ.scaler == "none") & summ.model.isin(["knn", "svr"])]
    add("C10", "Khi không scale, Target Encoding có MAE thấp nhất trong 4 encoder ở KNN và SVR",
        all(g.loc[g["mae_mean"].idxmin(), "encoder"] == "target" for _, g in m.groupby("model")))
    if conf is not None:
        add("C11", "Trên toàn bộ train, One-hot có MAE thấp nhất trong 4 encoder ở KNN, Ridge và Random Forest",
            all(g.loc[g["mae_full"].idxmin(), "encoder"] == "onehot" for _, g in conf.groupby("model")))
    lf = c["lfm"]
    if lf is not None:
        add("C12", "Log của biến đầu vào: MAE của KNN và SVR tăng ở mọi encoder, Ridge giảm ở mọi encoder, Random Forest đổi < 1,5%",
            (lf[lf.model.isin(["knn", "svr"])]["rel_change_pct"] > 0).all() and (lf[lf.model == "ridge"]["rel_change_pct"] < 0).all()
            and lf[lf.model == "rf"]["rel_change_pct"].abs().max() < 1.5)
    top = summ.sort_values("mae_mean").head(2)
    add("C13", "Chênh lệch MAE giữa cấu hình tốt nhất và tốt thứ hai nhỏ hơn độ lệch chuẩn giữa các fold của cấu hình tốt nhất",
        top["mae_mean"].iloc[1] - top["mae_mean"].iloc[0] < top["mae_std"].iloc[0])
    b = c["best"].set_index("model")["mae_mean"]
    add("C14", "Với giá gốc, Random Forest có MAE thấp nhất và Ridge cao nhất trong 4 mô hình (cấu hình tốt nhất của mỗi mô hình)",
        b.idxmin() == "rf" and b.idxmax() == "ridge")
    return out


# ------------------------------------------------------------------------------------------ sections
def section_design(c):
    return f"""## 2. Thiết kế thí nghiệm

**Câu hỏi nghiên cứu:** mã hóa biến phân loại và chuẩn hóa dữ liệu ảnh hưởng thế nào đến sai số dự đoán giá, số chiều và thời gian huấn luyện của bốn mô hình hồi quy.

| Thành phần | Các mức | Ghi chú |
|---|---|---|
| Encoding | One-hot, Ordinal, Target, Frequency | One-hot: gom danh mục xuất hiện < 10 lần trong fold train vào một cột "infrequent". Ordinal: mã theo thứ tự chữ cái, danh mục lạ = −1. Target: `TargetEncoder` của sklearn (cross-fitting 5 fold + làm mượt). Frequency: tự cài đặt (`src/encoders.py`), danh mục lạ = 0 |
| Scaling | None, Standard, MinMax, Robust | Áp dụng cho biến số và cho các cột đã mã hóa số (Ordinal/Target/Frequency). Các cột One-hot (0/1) không scale |
| Mô hình | KNN, Ridge, SVR, Random Forest | Siêu tham số **cố định, chưa tinh chỉnh**: KNN k=10 (trọng số theo khoảng cách); Ridge α=1; SVR RBF C=10, ε=0,1 (target được chuẩn hóa bên trong SVR vì ε và C tính theo đơn vị của y); Random Forest 100 cây, `min_samples_leaf=2` |
| Biến đầu vào | 9 biến phân loại + 6 biến số | `brand`, `model`, `fuel_type`, `transmission`, `body_type`, `country_code`, `seller_type`, `drive_train`, `body_color`; `production_year`, `vehicle_age`, `mileage_km`, `power_kw`, `engine_cc`, `nr_seats`. Không có `model_version` (xem thí nghiệm riêng) |
| Thiếu dữ liệu | Trung vị (kèm cờ báo thiếu) cho biến số; nhãn `missing` cho biến phân loại | Học trong từng fold train |

**Năm nhóm thí nghiệm** (tổng {c['n_configs']} cấu hình × 5 fold = {c['n_tasks']} lần huấn luyện trên tập con, cộng {c['n_confirm_tasks']} lần xác nhận trên toàn train):

1. **main:** 4 encoder × 4 scaler × 4 mô hình = 64 cấu hình, target là giá gốc (EUR).
2. **log_target:** biến đổi log (`log1p`) áp dụng cho **giá** (target), 4 encoder × 4 mô hình, scaler Standard; dự đoán được đổi ngược bằng `expm1` nên mọi chỉ số vẫn tính theo EUR.
3. **log_features:** `log1p` áp dụng cho ba **biến đầu vào** lệch phải (`mileage_km`, `power_kw`, `engine_cc`) trước khi điền thiếu và scale, 4 encoder × 4 mô hình, scaler Standard, giá gốc. Hai thí nghiệm 2 và 3 cùng làm nên thí nghiệm riêng về log transformation.
4. **model_version:** thêm `model_version` ({c['mv_unique']:,} danh mục) vào 9 biến phân loại, 4 encoder × 4 mô hình, scaler Standard.
5. **full_train_confirm:** chạy lại 4 encoder trên **toàn bộ train** ({c['n_train']:,} dòng) cho KNN, Ridge, Random Forest để kiểm tra kết luận có giữ nguyên khi tăng dữ liệu không (SVR bỏ qua vì độ phức tạp O(n²)).

**Đánh giá:** GroupKFold 5 fold trên tập train, nhóm = `group_id` (cùng một xe không nằm ở cả hai phía của fold). Lưới chính chạy trên **tập con {c['dev_rows']:,} dòng** của train (lấy theo nhóm xe nguyên vẹn, seed 42) để SVR chạy được trong thời gian hợp lý và để mọi mô hình so sánh trên cùng dữ liệu. **Tập test không được đọc ở bất kỳ bước nào của tuần 2** (có unit test kiểm tra). Mỗi lần chạy lưu "chữ ký" (dữ liệu, số fold, seed, siêu tham số, phiên bản sklearn) và từ chối ghi tiếp vào kết quả cũ nếu chữ ký khác, để không trộn kết quả khác điều kiện.

**Chỉ số:** MAE (chỉ số chính, vì giá lệch rất mạnh), RMSE, R², MedAPE (trung vị sai số phần trăm), số chiều đầu vào mô hình, thời gian huấn luyện. RMSE và R² bị chi phối bởi một số ít xe cực đắt nên dao động mạnh giữa các fold; vì vậy kết luận dựa trên MAE/MedAPE và trên **so sánh cặp theo fold** (đếm số fold trong 5 fold mà phương pháp A tốt hơn B). 5 fold chưa đủ cho kiểm định thống kê chính thức nên báo cáo không đưa p-value. Thời gian huấn luyện được đo khi nhiều tiến trình chạy song song nên chỉ dùng để so sánh tương đối (con số tuyệt đối thay đổi theo tải máy).

**Chống rò rỉ:** imputer, scaler, encoder (kể cả Target Encoding) là các bước trong `Pipeline`, chỉ `fit` trên phần train của từng fold. Ghi chú: cross-fitting bên trong `TargetEncoder` chia ngẫu nhiên chứ không theo nhóm xe, nên các tin trùng xe trong cùng fold train có thể khiến mã hóa của train hơi lạc quan; ảnh hưởng này chỉ nằm trong phần train, không chạm vào fold đánh giá.
"""


def section_main(c):
    best = _lab(c["best"])
    b = best[["model", "encoder", "scaler", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "medape_pct_mean", "n_features_out", "fit_seconds_mean"]]
    b = b.rename(columns={"mae_mean": "MAE", "mae_std": "MAE std (fold)", "rmse_mean": "RMSE", "r2_mean": "R²", "medape_pct_mean": "MedAPE %",
                          "n_features_out": "số cột", "fit_seconds_mean": "fit (s)"})
    s = _lab(c["std_tab"])[["model", "encoder", "mae_mean", "rmse_mean", "r2_mean", "medape_pct_mean", "n_features_out", "fit_seconds_mean"]]
    s = s.rename(columns={"mae_mean": "MAE", "rmse_mean": "RMSE", "r2_mean": "R²", "medape_pct_mean": "MedAPE %", "n_features_out": "số cột", "fit_seconds_mean": "fit (s)"})
    return f"""## 3. Kết quả chính (giá gốc, 64 cấu hình)

**Cấu hình tốt nhất của từng mô hình (theo MAE trung bình 5 fold):**

{md_table(b)}

![Cấu hình tốt nhất của từng mô hình](figures/week2_best_models.png)

**Toàn bộ 4 encoder với scaler Standard** (cùng điều kiện cho cả bốn mô hình):

{md_table(s.sort_values(['model', 'MAE']))}

![MAE theo encoder × scaler](figures/week2_heatmap_mae.png)

Các heatmap tương ứng cho MedAPE, RMSE và R²: `figures/week2_heatmap_medape.png`, `week2_heatmap_rmse.png`, `week2_heatmap_r2.png`. Bảng đầy đủ 64 cấu hình: `tables/week2_main_grid_ranked.csv`.
"""


def section_encoding(c):
    pe = _lab(c["pe"])
    pe["folds_better"] = pe["folds_better"].astype(str) + " / " + pe["n"].astype(str)
    p = pe[["model", "level", "mean_diff", "rel_change_pct", "folds_better"]].rename(
        columns={"level": "encoder", "mean_diff": "ΔMAE so với One-hot (EUR)", "rel_change_pct": "Δ%", "folds_better": "số fold tốt hơn One-hot"})
    return f"""## 4. Ảnh hưởng của encoding

**So sánh cặp theo fold với One-hot** (scaler Standard; Δ > 0 nghĩa là encoder đó **tệ hơn** One-hot; cột cuối đếm số fold mà encoder đó tốt hơn One-hot):

{md_table(p.sort_values(['model', 'encoder']))}

**Số chiều và thời gian:** One-hot tạo {c['dims_onehot']} cột trên tập con ({c['dims_onehot_full']} cột trên toàn train), ba encoder còn lại chỉ {c['dims_other']} cột (9 biến phân loại + 6 biến số + các cột cờ báo thiếu của biến số). Thời gian huấn luyện của Random Forest tăng mạnh với One-hot (ma trận thưa nhiều cột) – xem biểu đồ.

![Số chiều và thời gian huấn luyện](figures/week2_dims_time.png)
"""


def section_scaling(c):
    ps = c["ps"]
    g = ps.groupby(["model", "level"]).agg(rel=("rel_change_pct", "mean"), better=("folds_better", "sum"), n=("n", "sum")).reset_index()
    g = _lab(g)
    g["cases"] = g["better"].astype(str) + " / " + g["n"].astype(str)
    g = g.rename(columns={"level": "scaler", "rel": "Δ MAE trung bình so với No scaling (%)", "cases": "số trường hợp tốt hơn No scaling (encoder × fold)"})
    return f"""## 5. Ảnh hưởng của scaling

Mỗi dòng gộp 4 encoder × 5 fold = 20 trường hợp so sánh cặp với No scaling.

{md_table(g[['model', 'scaler', 'Δ MAE trung bình so với No scaling (%)', 'số trường hợp tốt hơn No scaling (encoder × fold)']])}

![Ảnh hưởng của scaler](figures/week2_scaler_effect.png)
"""


def _paired_tab(x, a_name, b_name):
    x = _lab(x)
    x["folds_b_better"] = x["folds_b_better"].astype(str) + " / " + x["folds"].astype(str)
    return x[["model", "encoder", "a_mean", "b_mean", "rel_change_pct", "folds_b_better"]].rename(
        columns={"a_mean": a_name, "b_mean": b_name, "rel_change_pct": "Δ%", "folds_b_better": "số fold tốt hơn"}).sort_values(["model", "encoder"])


def section_log_target(c):
    return f"""## 6. Log transformation, phần 1: log của giá (target)

Scaler Standard cho cả hai phía; Δ% < 0 nghĩa là log giá **tốt hơn** giá gốc.

**MAE (EUR):**

{md_table(_paired_tab(c['lm'], 'giá gốc', 'log giá'))}

**RMSE (EUR):**

{md_table(_paired_tab(c['lr'], 'giá gốc', 'log giá'))}

![Hiệu ứng của log giá](figures/week2_log_target_effect.png)
"""


def section_log_features(c):
    if c["lfm"] is None:
        return "## 7. Log transformation, phần 2: log của biến đầu vào\n\n_Chưa chạy._\n"
    return f"""## 7. Log transformation, phần 2: log của biến đầu vào lệch

`log1p` áp dụng cho `mileage_km`, `power_kw`, `engine_cc` (giá gốc, scaler Standard cho cả hai phía). Δ% < 0 là log đầu vào **tốt hơn**.

**MAE (EUR):**

{md_table(_paired_tab(c['lfm'], 'không log', 'log đầu vào'))}

**RMSE (EUR):**

{md_table(_paired_tab(c['lfr'], 'không log', 'log đầu vào'))}

![Hiệu ứng của log biến đầu vào](figures/week2_log_features_effect.png)
"""


def section_version(c):
    mv = _paired_tab(c["mv"], "MAE không có model_version", "MAE có model_version")
    d = c["dims_mv"]
    return f"""## 8. Thí nghiệm cardinality: thêm `model_version`

Scaler Standard, giá gốc. Δ% < 0 là có `model_version` tốt hơn.

{md_table(mv)}

Số cột sau mã hóa (One-hot): {d['main']} → {d['mv']} khi thêm `model_version` (do gom danh mục < 10 lần vào cột "infrequent"). Các encoder còn lại: {d['other_main']} → {d['other_mv']} cột.

![Hiệu ứng của model_version](figures/week2_model_version_effect.png)
"""


def section_confirm(c):
    conf = c["conf"]
    if conf is None:
        return "## 9. Xác nhận trên toàn bộ train\n\n_Chưa chạy hoặc chưa hoàn tất: `python -m src.experiments --confirm`._\n"
    t = _lab(conf)[["model", "encoder", "scaler", "mae_dev", "mae_full", "rank_dev", "rank_full", "rmse_full", "r2_full", "fit_seconds_full"]].rename(
        columns={"mae_dev": f"MAE ({c['dev_k']}k)", "mae_full": f"MAE ({c['full_k']}k)", "rank_dev": f"hạng ({c['dev_k']}k)", "rank_full": f"hạng ({c['full_k']}k)",
                 "rmse_full": f"RMSE ({c['full_k']}k)", "r2_full": f"R² ({c['full_k']}k)", "fit_seconds_full": "fit (s)"})
    same = int((conf["rank_dev"] == conf["rank_full"]).sum())
    best_same = all(g.loc[g["mae_dev"].idxmin(), "encoder"] == g.loc[g["mae_full"].idxmin(), "encoder"] for _, g in conf.groupby("model"))
    return f"""## 9. Xác nhận trên toàn bộ train ({c['n_train']:,} dòng)

Mỗi mô hình dùng scaler tốt nhất của nó trên tập con {c['dev_k']}k. Thứ hạng encoder (1 = MAE thấp nhất) giữ nguyên ở {same}/{len(conf)} dòng; encoder tốt nhất của từng mô hình {'trùng' if best_same else 'KHÁC'} giữa {c['dev_k']}k và {c['full_k']}k.

{md_table(t.sort_values(['model', f"MAE ({c['full_k']}k)"]))}
"""


def section_discussion(c):
    pe, ps, lm, lr, lfm, mv = (c[k] for k in ("pe", "ps", "lm", "lr", "lfm", "mv"))
    summ, conf = c["summ"], c["conf"]
    sc = ps.groupby(["model", "level"])["rel_change_pct"].mean().unstack()
    rf_sc = ps[ps.model == "rf"]["rel_change_pct"].abs().max()
    ridge_log = lm[lm.model == "ridge"]["rel_change_pct"]
    knn_rf_rmse = lr[lr.model.isin(["knn", "rf"]) & (lr.encoder != "target")]["rel_change_pct"]
    ord_mv = mv[(mv.encoder == "ordinal") & mv.model.isin(["knn", "rf", "svr"])]["rel_change_pct"]
    ridge_pe = pe[pe.model == "ridge"]
    eff_sc, eff_enc = c["eff_sc"], c["eff_enc"]
    top = summ.sort_values("mae_mean").head(2)
    bo = top.iloc[0]
    dk, fk = c["dev_k"], c["full_k"]

    if conf is not None:
        cf = conf.set_index(["model", "encoder"])
        r_oh = cf.loc[("ridge", "onehot"), "mae_full"]
        r_other = min(cf.loc[("ridge", e), "mae_full"] for e in OTHER_ENCS)
        rf_t = cf.loc[("rf", "onehot"), "fit_seconds_full"]
        rf_o = sum(cf.loc[("rf", e), "fit_seconds_full"] for e in OTHER_ENCS) / 3
        rf_gain = (1 - cf.loc[("rf", "onehot"), "mae_full"] / min(cf.loc[("rf", e), "mae_full"] for e in OTHER_ENCS)) * 100
        conf_text = (f"**Xác nhận trên toàn bộ train làm thay đổi kết luận về Ridge:** trên tập con {dk}k, Ridge gần như không phân biệt các encoder, "
                     f"nhưng trên {fk}k dòng One-hot đạt MAE {_eur(r_oh)} EUR so với {_eur(r_other)} EUR của encoder tốt nhất còn lại ({(r_oh / r_other - 1) * 100:+.0f}%). "
                     f"Một giải thích hợp lý (chưa kiểm chứng riêng) là với nhiều dữ liệu hơn, One-hot cho phép Ridge học hệ số riêng cho từng hãng/mẫu, còn Ordinal/Frequency/Target ép "
                     f"mỗi biến phân loại thành một số duy nhất nên mô hình tuyến tính không biểu diễn được. Với Random Forest, One-hot chỉ cải thiện MAE {rf_gain:.1f}% "
                     f"so với encoder tốt nhất còn lại nhưng thời gian huấn luyện lớn hơn khoảng {rf_t / rf_o:.0f} lần ({rf_t:,.0f} s so với ~{rf_o:,.0f} s mỗi fold; chỉ so sánh tương đối).")
    else:
        conf_text = "_Chưa có bước xác nhận trên toàn bộ train._"

    lf_text = ""
    if lfm is not None:
        m = lfm.groupby("model")["rel_change_pct"].mean()
        lf_text = (f"6. **Log của biến đầu vào lệch (mileage, power, engine)** → *kết quả trái chiều theo mô hình.* MAE trung bình thay đổi: Ridge {m['ridge']:+.1f}%, Random Forest {m['rf']:+.1f}%, "
                   f"SVR {m['svr']:+.1f}%, KNN {m['knn']:+.1f}%. Ridge hưởng lợi (quan hệ gần tuyến tính hơn sau log), Random Forest gần như không đổi (cây bất biến với biến đổi đơn điệu; chênh lệch nhỏ chỉ do giá trị điền thiếu thay đổi), "
                   "còn KNN và SVR tệ hơn: một giải thích hợp lý (chưa kiểm chứng riêng) là log làm thay đổi cấu trúc khoảng cách giữa các xe (xe km thấp bị tách xa nhau hơn xe km cao), "
                   "trong khi các mô hình dựa trên khoảng cách đã khai thác tốt thang đo gốc sau Standard.")

    return f"""## 10. Thảo luận và kiểm chứng giả thuyết đặt ra ở tuần 1

1. **"Log giá cải thiện mô hình tuyến tính"** → *được ủng hộ cho Ridge và SVR.* Với Ridge, log giá giảm MAE từ {abs(ridge_log.max()):.0f}% đến {abs(ridge_log.min()):.0f}% tùy encoder (5/5 fold). Nhưng với KNN và Random Forest, log giá làm **RMSE tệ hơn** {knn_rf_rmse.min():+.0f}% đến {knn_rf_rmse.max():+.0f}% (ngoại trừ Target Encoding) trong khi MAE đổi rất ít. Một giải thích hợp lý (chưa kiểm chứng riêng) là khi dự đoán trong không gian log rồi đổi ngược, các xe cực đắt bị dự đoán thấp hơn thực tế nên sai số ở đuôi giá tăng. Log giá không phải lựa chọn tốt hơn một cách tuyệt đối; hiệu quả phụ thuộc mô hình và chỉ số.
2. **"One-hot + Ridge tốt nhưng chiều lớn khi có model_version"** → *một phần: vế đầu đúng khi đủ dữ liệu, vế sau không xảy ra.* Trên tập con {dk}k với giá gốc, Ridge gần như không phân biệt được các encoder (chênh lệch tối đa {ridge_pe['rel_change_pct'].abs().max():.1f}%, mỗi encoder tốt hơn One-hot nhiều nhất {int(ridge_pe['folds_better'].max())}/5 fold); trên toàn bộ train One-hot vượt rõ rệt (mục 9). Khi thêm `model_version`, One-hot chỉ tăng từ {c['dims_mv']['main']} lên {c['dims_mv']['mv']} cột vì đã gom danh mục hiếm; MAE gần như không đổi (thay đổi tối đa {mv[(mv.encoder == 'onehot')]['rel_change_pct'].abs().max():.1f}% ở mọi mô hình). Điều này phù hợp với việc hầu hết giá trị `model_version` hiếm đến mức bị gom vào cột "infrequent" nên không đóng góp thông tin (giải thích hợp lý, chưa kiểm chứng riêng).
3. **"KNN/SVR nhạy với scaling"** → *được ủng hộ mạnh.* So với No scaling, Standard giảm MAE của KNN {abs(sc.loc['knn','standard']):.0f}% và của SVR {abs(sc.loc['svr','standard']):.0f}% (trung bình 4 encoder; tốt hơn ở 20/20 trường hợp mỗi mô hình). Một giải thích hợp lý (chưa kiểm chứng riêng) là khi không scale, `mileage_km` (hàng trăm nghìn) áp đảo khoảng cách của KNN và kernel của SVR.
4. **"Random Forest ít phụ thuộc scaling"** → *được ủng hộ.* Thay đổi MAE lớn nhất giữa các scaler chỉ {rf_sc:.2f}% (cây quyết định chỉ so sánh ngưỡng nên bất biến với biến đổi đơn điệu). Đây cũng là kiểm tra tính đúng đắn của pipeline.
5. **"Target Encoding chỉ có lợi nếu fit trong fold và làm mượt"** → *không có bằng chứng Target Encoding vượt One-hot trong lưới chính (và cả trên toàn train, mục 9):* với KNN, Random Forest và SVR, One-hot tốt hơn Target ở hầu hết fold. Target Encoding chỉ nổi bật khi KNN/SVR **không scale** (giá trị mã hóa cùng đơn vị với giá nên tự có thang đo lớn nhưng hợp lý). Cấu hình MAE thấp nhất toàn bộ thí nghiệm là {MODEL_LABEL[bo['model']]} + {ENC_LABEL[bo['encoder']]} + {SCALER_LABEL[bo['scaler']]} (nhóm `{bo['experiment']}`).
{lf_text}

**Kết luận về encoding:** One-hot cho MAE thấp hơn Ordinal và Frequency ở KNN, Random Forest và SVR trên tập con {dk}k (ở cả ba mô hình, Ordinal và Frequency không tốt hơn One-hot ở bất kỳ fold nào), và trên toàn bộ train One-hot là encoder tốt nhất của cả KNN, Ridge và Random Forest. {conf_text} Cái giá của One-hot là số cột ({c['dims_onehot']} trên {dk}k dòng, {c['dims_onehot_full']} trên toàn train, so với {c['dims_other']}) và thời gian huấn luyện. Ordinal và Frequency cho kết quả kém hơn, phù hợp với việc thứ tự giả và việc mất định danh danh mục đều gây hại cho các biến có nhiều danh mục như `model`. Với `model_version`, Ordinal tệ hơn rõ rệt ở KNN, Random Forest và SVR ({ord_mv.min():+.0f}% đến {ord_mv.max():+.0f}% MAE; 0/5 fold tốt hơn).

**Kết luận về scaling:** ảnh hưởng của scaler lớn hơn nhiều so với ảnh hưởng của encoder đối với KNN và SVR: chênh lệch MAE giữa scaler tốt nhất và tệ nhất là {eff_sc.loc['knn','spread_pct_of_min']:.0f}% (KNN) và {eff_sc.loc['svr','spread_pct_of_min']:.0f}% (SVR), trong khi giữa encoder tốt nhất và tệ nhất chỉ {eff_enc.loc['knn','spread_pct_of_min']:.0f}% và {eff_enc.loc['svr','spread_pct_of_min']:.0f}% (trung bình trên các mức của yếu tố còn lại; bảng `tables/week2_effect_scaler.csv`, `week2_effect_encoder.csv`). Ảnh hưởng của scaler nhỏ với Ridge ({eff_sc.loc['ridge','spread_pct_of_min']:.1f}%) và gần bằng 0 với Random Forest ({eff_sc.loc['rf','spread_pct_of_min']:.2f}%). Giữa Standard và Robust, khác biệt nhỏ hơn nhiều so với khác biệt giữa "có scale" và "không scale"; MinMax kém hơn hai scaler còn lại cho KNN và SVR, một giải thích hợp lý (chưa kiểm chứng riêng) là các biến có đuôi dài (mileage, power) bị nén bởi giá trị lớn nhất.

**Kết luận chung:** MAE thấp nhất toàn bộ là {_eur(bo['mae_mean'])} EUR; chênh lệch với cấu hình tốt thứ hai ({_eur(top['mae_mean'].iloc[1] - bo['mae_mean'])} EUR) so với độ lệch chuẩn giữa các fold ({_eur(bo['mae_std'])} EUR) cho thấy **không nên kết luận một cấu hình duy nhất là tốt nhất**; kết luận vững hơn là xếp hạng theo nhóm (giá gốc, siêu tham số cố định): Random Forest tốt nhất, SVR và KNN gần nhau ở giữa, Ridge kém nhất.
"""


def section_limits():
    return """## 11. Giới hạn và rủi ro

- Siêu tham số chưa được tinh chỉnh (k=10, α=1, C=10…); Ridge kém một phần vì quan hệ giá–đặc trưng phi tuyến và đuôi giá rất dài, một phần vì chưa chọn α. Xếp hạng giữa các mô hình có thể thay đổi sau khi tinh chỉnh (tuần 3).
- Lưới chính chạy trên tập con 20.000 dòng; phần xác nhận trên toàn train bỏ qua SVR. **Kết quả xác nhận cho thấy kết luận từ tập con không luôn bền** (Ridge, mục 9): mọi kết luận về encoding của Ridge cần dựa trên kết quả toàn train, và kết luận về SVR chưa được xác nhận trên dữ liệu lớn.
- Dữ liệu giá lệch cực mạnh, vài xe cực đắt làm RMSE và R² dao động giữa các fold; kết luận chính dựa trên MAE, MedAPE và so sánh cặp theo fold, không dựa trên RMSE.
- 5 fold chưa đủ cho kiểm định thống kê chính thức; báo cáo đếm số fold thay vì p-value.
- Mới thử một cách gom danh mục hiếm cho One-hot (ngưỡng 10); chưa thử gom cho các encoder khác. Log của biến đầu vào chỉ thử với ba biến lệch phải và scaler Standard.
- Các "giải thích hợp lý" trong báo cáo là giả thuyết, chưa được kiểm chứng bằng thí nghiệm riêng.
- Thời gian huấn luyện đo khi nhiều tiến trình chạy song song, chỉ dùng so sánh tương đối.
- Dữ liệu thiên về xe cao cấp châu Âu: kết luận về encoding/scaling cần kiểm lại trên dữ liệu Việt Nam.
- Cross-fitting của Target Encoding không chia theo nhóm xe (chỉ ảnh hưởng phần train).
- Chạy lại cùng cấu hình có thể lệch cỡ 1 phần triệu ở KNN (thứ tự cộng số thực khi chạy song song), không ảnh hưởng kết luận.
"""


def section_plan():
    return """## 12. Kế hoạch tuần 3

1. Tinh chỉnh siêu tham số bằng cross-validation lồng/GroupKFold trên train (k của KNN, α của Ridge, C/ε/γ của SVR, độ sâu/số cây của Random Forest).
2. Gom danh mục hiếm của `model_version` bằng thống kê học trên train; thử Target Encoding có làm mượt mạnh hơn.
3. Quyết định xử lý các xe giá cực cao (chỉ trên train).
4. Đánh giá cuối cùng **một lần** trên `test.parquet` với cấu hình đã chốt; phân tích phần dư theo hãng, quốc gia, khoảng giá.
5. Chạy lại phần xác nhận cho SVR trên dữ liệu lớn hơn (hoặc dùng xấp xỉ nhanh).
"""


def section_claims(claims):
    rows = pd.DataFrame([{"mã": i, "phát biểu trong báo cáo": t, "kiểm tra với bảng số liệu": "ĐÚNG" if ok else "KHÔNG CÒN ĐÚNG"} for i, t, ok in claims])
    return f"""## Phụ lục A. Tự kiểm chứng các phát biểu

Các phát biểu định tính ở mục 10 được máy tính kiểm tra lại với các bảng số liệu mỗi lần sinh báo cáo (`verify_claims` trong `src/reporting_week2.py`).

{md_table(rows)}
"""


# ------------------------------------------------------------------------------------------ main
def build_context() -> dict:
    summ = _t("week2_summary_all")
    meta = json.loads((R / "week2_run_meta.json").read_text(encoding="utf-8"))
    stats = json.loads((R / "stats.json").read_text(encoding="utf-8"))
    conf = _t("week2_confirm_full_train") if (T / "week2_confirm_full_train.csv").exists() else None
    std = summ[(summ["experiment"] == "main") & (summ["scaler"] == "standard")]
    mvd = summ[(summ["experiment"] == "model_version") & (summ["model"] == "ridge")].set_index("encoder")["n_features_out"]
    dims_onehot = int(std[std.encoder == "onehot"]["n_features_out"].iloc[0])
    dims_other = int(std[std.encoder == "ordinal"]["n_features_out"].iloc[0])
    n_train = int(stats["split"]["n_train"])
    ctx = dict(
        summ=summ, conf=conf, std_tab=std, best=_t("week2_best_per_model"),
        pe=_t("week2_paired_encoder_vs_onehot_standard"), ps=_t("week2_paired_scaler_vs_none"),
        lm=_t("week2_log_target_vs_raw_mae"), lr=_t("week2_log_target_vs_raw_rmse"), mv=_t("week2_model_version_vs_main_mae"),
        lfm=_t("week2_log_features_vs_raw_mae") if (T / "week2_log_features_vs_raw_mae.csv").exists() else None,
        lfr=_t("week2_log_features_vs_raw_rmse") if (T / "week2_log_features_vs_raw_rmse.csv").exists() else None,
        eff_sc=_t("week2_effect_scaler").set_index("model"), eff_enc=_t("week2_effect_encoder").set_index("model"),
        n_configs=int(summ.shape[0]), n_tasks=int(meta["n_result_rows"]), n_confirm_tasks=60 if conf is not None else 0,
        dev_rows=int(meta["n_dev_rows"]), dev_k=int(round(meta["n_dev_rows"] / 1000)), n_train=n_train, full_k=int(round(n_train / 1000)),
        mv_unique=int(next(r["n_unique"] for r in stats["cardinality"] if r["column"] == "model_version")),
        dims_onehot=dims_onehot, dims_other=dims_other,
        dims_onehot_full=int(conf["n_features"][conf["encoder"] == "onehot"].iloc[0]) if conf is not None else "?",
        dims_mv={"main": dims_onehot, "mv": int(mvd["onehot"]), "other_main": dims_other, "other_mv": int(mvd["ordinal"])},
    )
    return ctx


def write():
    c = build_context()
    claims = verify_claims(c)
    failed = [(i, t) for i, t, ok in claims if not ok]
    for i, t in failed:
        print(f"[CLAIM FAILED] {i}: {t}")
    banner = ("\n> **CẢNH BÁO:** có phát biểu ở mục 10 không còn đúng với số liệu hiện tại (xem Phụ lục A). Cần sửa phần diễn giải trước khi dùng báo cáo.\n"
              if failed else "")
    head = f"""# Báo cáo Tuần 2 – Encoding × Scaling × Mô hình hồi quy

_Mọi số liệu được tính từ `data/processed/train.parquet` và lưu ở `reports/tables/week2_*.csv`. Báo cáo sinh bằng `python -m src.reporting_week2`._
{banner}
## 1. Mục tiêu và phạm vi

Tuần 2 xây dựng và chạy bộ thí nghiệm so sánh bốn cách mã hóa biến phân loại, bốn cách chuẩn hóa (cùng hai thí nghiệm log transformation) và bốn mô hình hồi quy trên tập train đã chia ở tuần 1, rồi trả lời: cách mã hóa/chuẩn hóa nào ảnh hưởng đến sai số, số chiều và thời gian huấn luyện. **Không** dùng tập test, **không** tinh chỉnh siêu tham số (để tuần 3).
"""
    body = "\n".join([head, section_design(c), section_main(c), section_encoding(c), section_scaling(c), section_log_target(c),
                      section_log_features(c), section_version(c), section_confirm(c), section_discussion(c), section_limits(),
                      section_plan(), section_claims(claims)])
    (R / "week2_report.md").write_text(body, encoding="utf-8")
    print("wrote", R / "week2_report.md", "| claims:", len(claims), "| failed:", len(failed))
    return failed


if __name__ == "__main__":
    write()
