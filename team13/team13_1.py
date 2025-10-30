# ============================================
# Step 0：日本語フォント設定（文字化け防止）
# ============================================
!apt-get -y install fonts-ipafont-gothic > /dev/null

import subprocess
subprocess.run(["fc-cache", "-fv"], check=True)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import holidays
import os

# --- フォント自動検出 ---
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
if not font_candidates:
    raise FileNotFoundError("IPAフォントが見つかりません。ランタイムを再起動して再実行してください。")
font_path = font_candidates[0]
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False
print("✅ 使用中フォント名:", prop.get_name())


# ============================================
# Step 1：サンプルデータ生成
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

# カレンダー系特徴量
jp_holidays = holidays.Japan(years=[2024])
df["hour"] = df["datetime"].dt.hour
df["dow"] = df["datetime"].dt.dayofweek
df["is_holiday"] = df["datetime"].dt.date.astype(object).isin(jp_holidays).astype(int)

print("✅ サンプルデータ作成完了")
display(df.head())


# ============================================
# Step 2：派生・ラグ特徴量の作成
# ============================================
df["gas_lag_24"] = df["gas_usage"].shift(24)
df["gas_lag_168"] = df["gas_usage"].shift(168)
df["temp_squared"] = df["temperature"] ** 2
df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"]
df["electricity_diff"] = df["electricity"].diff()

# 欠損削除
df = df.dropna().reset_index(drop=True)
print("✅ 特徴量作成完了")
display(df.head())


# ============================================
# Step 3：ベース＋特徴量ごとの効果比較
# ============================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# --- ベース特徴量 ---
base_features = ["temperature", "humidity", "wind_speed", "electricity"]

# --- 追加候補特徴量 ---
candidate_features = ["gas_lag_24", "gas_lag_168", "hour", "dow", "is_holiday"]

results = []

# --- ① ベースモデル（追加なし） ---
X_base = df[base_features]
y = df["gas_usage"]

X_train, X_valid, y_train, y_valid = train_test_split(X_base, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_valid)

rmse = mean_squared_error(y_valid, pred) ** 0.5
r2 = r2_score(y_valid, pred)

results.append({"追加特徴量": "（ベースのみ）", "RMSE": rmse, "R²": r2})

# --- ② 各特徴量を1つずつ追加して比較 ---
for feat in candidate_features:
    X_tmp = df[base_features + [feat]]
    y_tmp = df["gas_usage"]

    X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    rmse = mean_squared_error(y_valid, pred) ** 0.5
    r2 = r2_score(y_valid, pred)

    results.append({"追加特徴量": f"+ {feat}", "RMSE": rmse, "R²": r2})

# --- 結果表示 ---
results_df = pd.DataFrame(results)
print("=== 特徴量ごとの効果比較（ベース含む） ===")
display(results_df)

# --- グラフ表示（RMSE） ---
plt.figure(figsize=(8,4))
plt.bar(results_df["追加特徴量"], results_df["RMSE"])
plt.title("追加特徴量ごとのRMSE比較", fontproperties=prop)
plt.ylabel("RMSE（小さいほど良い）", fontproperties=prop)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- グラフ表示（R²） ---
plt.figure(figsize=(8,4))
plt.bar(results_df["追加特徴量"], results_df["R²"])
plt.title("追加特徴量ごとのR²比較", fontproperties=prop)
plt.ylabel("R²（1に近いほど良い）", fontproperties=prop)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
