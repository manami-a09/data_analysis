# １週間のガス使用量予測


# ============================================
# Step 6：将来データでの予測（未来1週間のガス使用量）
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

# --- 未来データ作成 ---
future_start = pd.Timestamp("2024-04-01 00:00")
future_end   = pd.Timestamp("2024-04-07 23:00")
future_index = pd.date_range(future_start, future_end, freq="h")

hours_future = np.arange(len(future_index))

# 気象などは「想定シナリオ」でランダム生成（本来は気象予報データなどを使用）
temp_future = 10 + 10*np.sin(2*np.pi*hours_future/(24*90)) + np.random.normal(0, 2, len(hours_future))
humidity_future = np.clip(60 + 20*np.sin(2*np.pi*hours_future/24) + np.random.normal(0,5,len(hours_future)), 20, 100)
wind_future = np.clip(np.random.gamma(2.0,1.2,len(hours_future)), 0, None)
weather_future = np.random.choice(["晴れ","曇り","雨"], len(hours_future), p=[0.6,0.3,0.1])
elec_future = 300 + 5*np.abs(temp_future-20) + np.random.normal(0,10,len(hours_future))

future_df = pd.DataFrame({
    "datetime": future_index,
    "temperature": temp_future,
    "humidity": humidity_future,
    "wind_speed": wind_future,
    "weather": weather_future,
    "electricity": elec_future
})

# --- カレンダー系特徴量 ---
jp_holidays = holidays.Japan(years=[2024])
future_df["hour"] = future_df["datetime"].dt.hour
future_df["dow"] = future_df["datetime"].dt.dayofweek
future_df["is_holiday"] = future_df["datetime"].dt.date.astype(object).isin(jp_holidays).astype(int)

# --- 過去データのラグを再現（未来データ単独では作れないため直近値を使用） ---
future_df["gas_lag_24"] = df["gas_usage"].iloc[-24:].values.repeat(len(future_df)//24 + 1)[:len(future_df)]
future_df["gas_lag_168"] = df["gas_usage"].iloc[-168:].values.repeat(len(future_df)//168 + 1)[:len(future_df)]

# --- 使用する特徴量を合わせる ---
X_future = future_df[features]

# --- 予測実行 ---
y_future_pred = best_model.predict(X_future)
future_df["predicted_gas_usage"] = y_future_pred

# ============================================
# 結果可視化
# ============================================
plt.figure(figsize=(12,5))
plt.plot(future_df["datetime"], future_df["predicted_gas_usage"], label="予測ガス使用量", color="red")
plt.title("未来1週間のガス使用量予測", fontproperties=prop)
plt.xlabel("日時", fontproperties=prop)
plt.ylabel("予測ガス使用量", fontproperties=prop)
plt.legend(prop=prop)
plt.grid(True)
plt.tight_layout()
plt.show()

display(future_df.head(10))
