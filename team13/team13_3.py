# パラメータチューニング


# ============================================
# Step 5：LightGBM パラメータチューニング（改善版）
# ============================================

!pip install optuna -q

import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import pandas as pd

# --- フォント設定（文字化け防止） ---
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
font_path = font_candidates[0]
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False

print("✅ Step5 改善版 - LightGBM × Optuna 実行開始")

# ============================================
# ① 特徴量・目的変数の設定（Step4で使用した df を前提）
# ============================================
# 👉 派生特徴量のうちノイズ傾向の強いものを削除
features = [
    "temperature", "humidity", "wind_speed", "electricity",
    "gas_lag_24", "gas_lag_168", "hour", "dow", "is_holiday"
]
target = "gas_usage"

X = df[features]
y = df[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ============================================
# ② Optuna による安定探索（過学習抑制・探索範囲調整）
# ============================================

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 31, 120),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
        "n_estimators": 1000
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50)]  # 早期終了で過学習防止
    )

    y_pred = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred) ** 0.5
    return rmse

# --- 試行回数を拡大（精度安定化） ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=60, show_progress_bar=True)

# ============================================
# ③ ベストパラメータで再学習
# ============================================
best_params = study.best_params
print("\n✅ 最適パラメータ:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_valid)

rmse_best = mean_squared_error(y_valid, y_pred_best) ** 0.5
r2_best = r2_score(y_valid, y_pred_best)

print(f"\n【最適モデル結果】RMSE={rmse_best:.2f}, R²={r2_best:.3f}")

# ============================================
# ④ Step4との比較（RMSE / R²）
# ============================================

# Step4（前回の結果）
rmse_step4 = 9.15
r2_step4 = 0.893

compare_df = pd.DataFrame({
    "モデル": ["Step4（基本）", "Step5（改善チューニング）"],
    "RMSE": [rmse_step4, rmse_best],
    "R²": [r2_step4, r2_best]
})
display(compare_df)

# --- 可視化（棒グラフで比較） ---
fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()
width = 0.35

ax1.bar([0 - width/2, 1 - width/2], compare_df["RMSE"], width, label="RMSE", color="#6fa8dc")
ax2.bar([0 + width/2, 1 + width/2], compare_df["R²"], width, label="R²", color="#93c47d")

ax1.set_ylabel("RMSE（小さいほど良い）", fontproperties=prop)
ax2.set_ylabel("R²（大きいほど良い）", fontproperties=prop)
plt.xticks([0,1], compare_df["モデル"], fontproperties=prop)
plt.title("Step4 vs Step5（改善後）モデル比較", fontproperties=prop)
fig.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ============================================
# ⑤ 特徴量重要度の再確認
# ============================================
importance = pd.DataFrame({
    "特徴量": X_train.columns,
    "重要度": best_model.feature_importances_
}).sort_values("重要度", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(importance["特徴量"][::-1], importance["重要度"][::-1])
plt.title("最適化後の特徴量重要度（LightGBM + Optuna 改善版）", fontproperties=prop)
plt.xlabel("重要度スコア", fontproperties=prop)
plt.tight_layout()
plt.show()

display(importance)
