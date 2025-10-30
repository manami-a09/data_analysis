# LightGBMで特徴量重要度を検証


# ============================================
# Step 4：LightGBMで特徴量重要度を検証
# ============================================

# --- 必要ライブラリ ---
!pip install lightgbm -q

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import holidays

# --- フォント設定（再利用） ---
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
font_path = font_candidates[0]
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False
print("✅ 使用中フォント:", prop.get_name())

# ============================================
# データ生成（Step1〜2の再掲）
# ============================================
start = pd.Timestamp("2024-01-01 00:00")
end   = pd.Timestamp("2024-03-31 23:00")
dt_index = pd.date_range(start, end, freq="h")

hours = np.arange(len(dt_index))
temp = 10 + 10*np.sin(2*np.pi*hours/(24*90)) + np.random.normal(0,2,len(hours))
humidity = np.clip(60 + 20*np.sin(2*np.pi*hours/24) + np.random.normal(0,5,len(hours)), 20, 100)
wind = np.clip(np.random.gamma(2.0,1.2,len(hours)), 0, None)
weather = np.random.choice(["晴れ","曇り","雨"], len(hours), p=[0.6,0.3,0.1])
elec = 300 + 5*np.abs(temp-20) + np.random.normal(0,10,len(hours))
gas = 200 + 0.5*elec + 5*np.maximum(15-temp, 0) + 10*(weather=="雨") + np.random.normal(0,8,len(hours))

df = pd.DataFrame({
    "datetime": dt_index,
    "temperature": temp,
    "humidity": humidity,
    "wind_speed": wind,
    "weather": weather,
    "electricity": elec,
    "gas_usage": gas
})

jp_holidays = holidays.Japan(years=[2024])
df["hour"] = df["datetime"].dt.hour
df["dow"] = df["datetime"].dt.dayofweek
df["is_holiday"] = df["datetime"].dt.date.astype(object).isin(jp_holidays).astype(int)

df["gas_lag_24"] = df["gas_usage"].shift(24)
df["gas_lag_168"] = df["gas_usage"].shift(168)
df["temp_squared"] = df["temperature"] ** 2
df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"]
df["electricity_diff"] = df["electricity"].diff()

df = df.dropna().reset_index(drop=True)

print("✅ データ生成完了")
display(df.head())

# ============================================
# LightGBMモデル構築と特徴量重要度検証
# ============================================

# --- 使用する特徴量 ---
features = [
    "temperature", "humidity", "wind_speed", "electricity",
    "gas_lag_24", "gas_lag_168", "hour", "dow", "is_holiday",
    "temp_squared", "temp_wind_interaction", "electricity_diff"
]
target = "gas_usage"

X = df[features]
y = df[target]

# --- データ分割 ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- LightGBM設定 ---
params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "regression",
}

model = lgb.LGBMRegressor(**params)
callbacks = [lgb.early_stopping(100, verbose=False)]

# --- 学習 ---
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=callbacks
)

# --- 精度確認 ---
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred) ** 0.5
r2 = r2_score(y_valid, y_pred)

print(f"【LightGBM結果】RMSE={rmse:.2f}, R²={r2:.3f}")

# ============================================
# 特徴量重要度の可視化
# ============================================
importance = pd.DataFrame({
    "特徴量": X_train.columns,
    "重要度": model.feature_importances_
}).sort_values("重要度", ascending=False)

plt.figure(figsize=(8,5))
plt.barh(importance["特徴量"][::-1], importance["重要度"][::-1])
plt.title("特徴量重要度（LightGBM）", fontproperties=prop)
plt.xlabel("重要度スコア", fontproperties=prop)
plt.tight_layout()
plt.show()

display(importance)
