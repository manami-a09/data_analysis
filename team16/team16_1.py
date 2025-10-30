# ============================================
# Step 0：日本語フォント設定（文字化け防止）
# ============================================
!apt-get -y install fonts-ipafont-gothic > /dev/null
import subprocess, matplotlib.pyplot as plt, matplotlib as mpl, matplotlib.font_manager as fm
subprocess.run(["fc-cache", "-fv"], check=True)
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
font_path = font_candidates[0]; fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path); mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False; print("✅ 使用中フォント:", prop.get_name())

# ============================================
# Step 1：ライブラリ読み込み
# ============================================
import pandas as pd, numpy as np, holidays
from sklearn.cluster import KMeans
import statsmodels.api as sm

# ============================================
# Step 2：データ作成（3ヶ月分の模擬データ）
# ============================================
np.random.seed(0)
dates = pd.date_range('2024-01-01', '2024-03-31 23:00', freq='H')
n_cust = 150
custs = [f"C{i:03d}" for i in range(n_cust)]

data = []
for c in custs:
    base = np.random.uniform(1.5, 3.0)
    pattern = np.random.choice(['昼型','夜型','平準型'])
    for t in dates:
        hour = t.hour
        if pattern == '昼型':
            load = base + 1.5*np.sin((hour-6)/24*2*np.pi)
        elif pattern == '夜型':
            load = base + 1.2*np.sin((hour+6)/24*2*np.pi)
        else:
            load = base
        load += np.random.normal(0,0.2)
        data.append([c, t, load, pattern])
df = pd.DataFrame(data, columns=['顧客ID','日時','使用量(kWh)','元タイプ'])

# 外生要因（気温・市場価格・祝日）
jp_holidays = holidays.Japan(years=[2024])
df['時刻'] = df['日時'].dt.hour
df['祝日'] = df['日時'].apply(lambda x: (x in jp_holidays) or x.weekday()==6)
df['気温(℃)'] = 10 + 10*np.sin((df['日時'].dt.dayofyear/365)*2*np.pi)
df['市場価格(JEPX)'] = 15 + 5*np.sin((df['日時'].dt.dayofyear/365)*2*np.pi + np.pi/4)

# ============================================
# Step 3：クラスタリング（負荷パターン分類）
# ============================================
pivot = df.groupby(['顧客ID','時刻'])['使用量(kWh)'].mean().unstack().fillna(0)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(pivot)
pivot['クラスタ'] = kmeans.labels_

# --- クラスタの特徴を自動判定 ---
cluster_profiles = pivot.groupby('クラスタ').mean()
cluster_labels = {}
for c in cluster_profiles.index:
    profile = cluster_profiles.loc[c].values[:24]
    day_mean = np.mean(profile[8:18])       # 昼間(8〜17時)
    night_mean = np.mean(profile[20:24])    # 夜間(20〜23時)
    early_mean = np.mean(profile[0:6])      # 深夜(0〜5時)
    ratio_day = day_mean / np.mean(profile)
    ratio_night = (night_mean + early_mean) / 2 / np.mean(profile)

    if ratio_day > 1.1 and ratio_day > ratio_night:
        label = "昼間ピーク型（オフィス・商業系）"
    elif ratio_night > 1.1:
        label = "夜間ピーク型（家庭・工場系）"
    else:
        label = "平準型（24h稼働・一定負荷）"
    cluster_labels[c] = label
pivot['クラスタ説明'] = pivot['クラスタ'].map(cluster_labels)

# ============================================
# Step 4：価格感応度（市場価格弾力性）算出
# ============================================
coef_list = []
for cid in df['顧客ID'].unique():
    sub = df[df['顧客ID']==cid]
    if sub['市場価格(JEPX)'].nunique() > 10:
        X = sm.add_constant(sub[['市場価格(JEPX)','気温(℃)']])
        y = sub['使用量(kWh)']
        try:
            res = sm.OLS(y,X).fit()
            coef_list.append([cid, res.params['市場価格(JEPX)']])
        except:
            coef_list.append([cid, np.nan])
    else:
        coef_list.append([cid, np.nan])
coef_df = pd.DataFrame(coef_list, columns=['顧客ID','市場価格弾力性'])

'''
| 弾力性の値          | 意味      | 解釈例                     |
| -------------- | ------- | ----------------------- |
| **-0.05〜-0.1** | 弱い負の相関  | 価格が上がると少し使用を減らす（やや敏感）   |
| **≈ 0**        | ほぼ影響なし  | 価格が上がっても使用量はほぼ変わらない     |
| **> 0**        | 正の相関（稀） | 特殊ケース（暖房など、価格上昇と同時に需要増） |

'''
# ============================================
# Step 5：昼夜平均・収益シミュレーション
# ============================================
df['時間帯'] = np.where(df['時刻'].between(8,18), '昼間', '夜間')
avg_usage = df.groupby(['顧客ID','時間帯'])['使用量(kWh)'].mean().unstack()

# --- プラン設定（A,B,C 各プラン名＋単価） ---
tariff = {
    "A：スタンダード型": {"昼間":28, "夜間":22},    # 時間帯差が小さい
    "B：ピークシフト型（夜安）": {"昼間":30, "夜間":20},  # 昼高・夜安
    "C：フラット型": {"昼間":26, "夜間":26}          # 均一単価
}
revenue = {}
for plan, price in tariff.items():
    revenue[plan] = (avg_usage['昼間']*price['昼間'] + avg_usage['夜間']*price['夜間'])*90  # 3ヶ月分想定
revenue_df = pd.DataFrame(revenue).reset_index()

# ============================================
# Step 6：統合データ作成（分析結果まとめ）
# ============================================
summary = (
    pivot[['クラスタ','クラスタ説明']]
    .reset_index()
    .merge(avg_usage.reset_index(), on='顧客ID')
    .merge(coef_df, on='顧客ID')
    .merge(revenue_df, on='顧客ID')
)
summary.rename(columns={
    '昼間':'昼間平均(kWh)',
    '夜間':'夜間平均(kWh)'
}, inplace=True)

# 表示
print("\n=== 顧客別 分析サマリ（提案用データ） ===")
display(summary.head(10))

# ============================================
# Step 7：クラスタ別傾向の可視化
# ============================================
plt.figure(figsize=(8,4))
for c in cluster_profiles.index:
    plt.plot(range(24), cluster_profiles.loc[c].values[:24], label=f"{cluster_labels[c]}")
plt.title("クラスタ別 平均負荷曲線")
plt.xlabel("時刻"); plt.ylabel("使用量(kWh)")
plt.legend(); plt.grid(True)
plt.show()

# ============================================
# Step 8：クラスタ別 平均収益比較
# ============================================
plan_summary = summary.groupby('クラスタ説明')[
    ['A：スタンダード型','B：ピークシフト型（夜安）','C：フラット型']
].mean()
print("\n=== クラスタ別 平均収益（料金案比較） ===")
display(plan_summary)
