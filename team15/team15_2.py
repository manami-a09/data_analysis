#ページタイトル、検索語をトッピック分類し、失注率を算出 　※コード1実行後


# ============================================
# Step 0：日本語フォント設定（文字化け防止）
# ============================================
!apt-get -y install fonts-ipafont-gothic fonts-ipafont-mincho > /dev/null

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import subprocess, os

# フォントキャッシュ更新
subprocess.run(["fc-cache", "-fv"], check=True)

# IPAフォントを指定（存在チェックつき）
font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
if not os.path.exists(font_path):
    font_path = "/usr/share/fonts/truetype/ipafont/ipagp.ttf"

fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 12
print("✅ 使用中フォント:", prop.get_name())


# ============================================
# Step 1：環境セットアップ
# ============================================
!pip -q install pandas numpy

import pandas as pd
import numpy as np
np.random.seed(42)


# ============================================
# Step 2：GA4セッション風データの作成
# ============================================
N = 5000
sessions = pd.DataFrame({
    "session_id": [f"s{i}" for i in range(N)],
    "user_pseudo_id": np.random.randint(1000, 1500, size=N),
    "source_medium": np.random.choice(
        ["google/organic", "google/cpc", "email/newsletter",
         "referral/price-compare", "direct/none"],
        size=N, p=[0.4,0.2,0.15,0.15,0.1]),
    "device_category": np.random.choice(["desktop", "mobile"], size=N, p=[0.55,0.45]),
    "pageviews": np.random.poisson(3, size=N)+1,
    "events": np.random.poisson(5, size=N)+1,
    "engaged": np.random.choice([0,1], size=N, p=[0.4,0.6]),
    "landing": np.random.choice(["/","/plans","/campaign","/faq","/cancel","/compare"], size=N),
    "last_page": np.random.choice(["/apply","/contact","/faq","/compare","/cancel"], size=N),
})

# 検索語・ページタイトル追加
search_terms = [None,"料金","解約","違約金","乗換","引越し","深夜料金","ecoプラン","工事費","シミュレーター","ポイント"]
sessions["search_term"] = np.where(np.random.rand(N)<0.4, np.random.choice(search_terms, size=N), None)

title_pool = [
    "電気料金シミュレーター","解約手続きの流れ","よくある質問(FAQ)","他社との比較",
    "キャンペーン詳細","お申込み(オンライン)","ポイント付与について","夜間割引の仕組み",
    "引越し時の手続き","工事費と工期","法人向けプラン"
]
sessions["page_title"] = np.random.choice(title_pool, size=N)

sessions.to_csv("ga4_sessions.csv", index=False)
print("✅ ga4_sessions.csv を作成しました")
print(sessions.head())


# ============================================
# Step 3：CRMデータ作成（成約/失注）
# ============================================

# ユニークな顧客一覧 を取得
sessions["user_pseudo_id"] = sessions["user_pseudo_id"].astype(int)
crm = sessions[["user_pseudo_id"]].drop_duplicates().copy()

# 行動傾向を集計
user_flags = (
    sessions.groupby("user_pseudo_id")
    .agg({
        "source_medium": lambda s: (s == "referral/price-compare").mean(),
        "search_term": lambda s: s.fillna("").str.contains("解約|違約金|工事費|比較").mean(),
    })
    .rename(columns={"source_medium": "pc_ref", "search_term": "neg_terms"})
)

'''
| 列名            | 内容                                        |
| :------------ | :---------------------------------------- |
| **pc_ref**    | 価格比較サイト（`referral/price-compare`）からのアクセス率 |
| **neg_terms** | 検索語の中に「解約」「違約金」「比較」など“ネガティブ系”が含まれる割合      |

'''

# 失注確率とフラグ
base_p = 0.3
prob = base_p + 0.5 * user_flags["pc_ref"] + 0.5 * user_flags["neg_terms"]

'''
| 要素                        | 内容            | 例                      | 重み（影響度） |
| :------------------------ | :------------ | :--------------------- | :-----: |
| `base_p`                  | 基本の離脱率        | どんな顧客でも離脱するリスク         |  +0.30  |
| `user_flags["pc_ref"]`    | 比較サイト経由の割合    | 比較サイトから来てる人ほど離脱しやすい    |  +0.5倍  |
| `user_flags["neg_terms"]` | 解約・違約金などの検索割合 | ネガティブワードを検索する人ほど離脱しやすい |  +0.5倍  |

'''

prob = prob.clip(0, 0.95)

'''
失注確率が 1.3 など「100%を超えてしまう」人が出るとおかしいので、
　0〜0.95（＝0%〜95%）の範囲に丸める 処理
'''


crm = crm.merge(prob.rename("lost_prob"), left_on="user_pseudo_id", right_index=True, how="left")
'''
prob にはユーザーごとの失注確率（0〜0.95）が入っています。
それを user_pseudo_id をキーにして、crm テーブルにくっつけています。
| user_pseudo_id | lost_prob |
| :------------- | :-------: |
| 1001           |    0.70   |
| 1002           |    0.35   |
| 1003           |    0.55   |

'''

crm["lost_flag"] = (np.random.rand(len(crm)) < crm["lost_prob"]).astype(int)
'''
やっていること：
np.random.rand(len(crm)) で、0〜1の乱数をユーザー数ぶん生成します。
各人の lost_prob（失注確率）と比べて、
乱数が lost_prob より小さい → 失注（1）
乱数が lost_prob より大きい → 成約（0）
.astype(int) で True/False を 1/0 に変換しています。

| user_pseudo_id | lost_prob |  乱数  |        判定       | lost_flag |
| :------------- | :-------: | :--: | :-------------: | :-------: |
| 1001           |    0.70   | 0.45 | 0.45 < 0.70 → ✅ |     1     |
| 1002           |    0.35   | 0.62 | 0.62 > 0.35 → ❌ |     0     |
| 1003           |    0.55   | 0.49 | 0.49 < 0.55 → ✅ |     1     |

乱数を生成する理由
🎲「サイコロを振るようなもの」です！
ある人は「7割の確率で離脱しそう」→ 10面ダイスを振って 1〜7 が出たら離脱
ある人は「3割の確率で離脱しそう」→ 10面ダイスを振って 1〜3 が出たら離脱
だから「乱数を投げる＝確率に従ってランダムに決める」という意味なんです。

'''


crm.to_csv("crm_data.csv", index=False)
print("✅ crm_data.csv を作成しました")
print(crm.head())
print("\n全ユーザー数:", len(crm))
print("失注率:", round(crm["lost_flag"].mean(), 3))


# ============================================
# Step 4：失注率分析（可視化）
# ============================================
data = sessions.merge(crm, on="user_pseudo_id", how="left")

# ① 流入別失注率
lost_by_source = data.groupby("source_medium")["lost_flag"].mean().sort_values(ascending=False)
print("\n【流入チャネル別 失注率】\n", lost_by_source)

# ② ページ別失注率
lost_by_page = data.groupby("page_title")["lost_flag"].mean().sort_values(ascending=False)
print("\n【ページタイトル別 失注率TOP5】\n", lost_by_page.head(5))

# ③ 検索語別失注率
lost_by_search = data.groupby("search_term")["lost_flag"].mean().sort_values(ascending=False)
print("\n【検索語別 失注率TOP5】\n", lost_by_search.head(5))

# ④ グラフ表示（日本語フォント適用）
plt.figure(figsize=(8,4))
lost_by_source.plot(kind="bar", color="orange")
plt.title("流入チャネル別の失注率", fontproperties=prop)
plt.ylabel("失注率", fontproperties=prop)
plt.xlabel("流入チャネル", fontproperties=prop)
plt.xticks(rotation=45, fontproperties=prop)
plt.tight_layout()
plt.show()
