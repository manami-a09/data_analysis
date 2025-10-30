# ============================================
# 0) セットアップ
# ============================================
!pip install lightgbm scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import early_stopping, log_evaluation  # callbackをインポート

np.random.seed(42)

# ============================================
# 1) サンプルデータ作成
# ============================================
n = 3000
areas = np.random.choice(["東A","東B","西A","西B","中部"], size=n)
industries = np.random.choice(["製造","IT","小売","物流","医療"], size=n)
sizes = np.random.choice(["小規模","中規模","大規模"], size=n, p=[0.4,0.4,0.2])
competitor = np.random.choice(["競合X","競合Y","競合Z"], size=n)
months_since_loss = np.random.randint(1, 36, size=n)
prev_contract_value = np.random.gamma(2.0, 50000, size=n)
last_contact_days = np.random.randint(3, 120, size=n)

# 失注理由テキスト（日本語）
reasons = [
    "価格が高い", "サポートが遅い", "切替コストが高い", "品質問題", "対応が不親切",
    "他社が安い", "手続きが複雑", "請求に不満", "電力量の見積違い", "障害対応が遅い"
]
lost_reason_text = ["、".join(np.random.choice(reasons, size=np.random.randint(1,3), replace=False))
                    for _ in range(n)]

# 営業チーム（5チーム）と、それぞれの得意領域を仮定
teams = np.random.choice(["チームA","チームB","チームC","チームD","チームE"], size=n)
team_strength = {
    "チームA": {"industry":"製造", "area":"東A"},
    "チームB": {"industry":"IT", "area":"東B"},
    "チームC": {"industry":"小売", "area":"西A"},
    "チームD": {"industry":"物流", "area":"西B"},
    "チームE": {"industry":"医療", "area":"中部"},
}

# 擬似的な「奪回しやすさ」を作成
base = -1.0 + 0.02*(36-months_since_loss) + 0.000008*prev_contract_value - 0.005*last_contact_days
for i in range(n):
    if industries[i] == team_strength[teams[i]]["industry"]:
        base[i] += 0.4  # 得意業界補正
    if areas[i] == team_strength[teams[i]]["area"]:
        base[i] += 0.3  # 得意エリア補正

# sigmoidで確率化
p = 1/(1+np.exp(-base))
y = (np.random.rand(n) < p).astype(int)

df = pd.DataFrame({
    "area": areas,
    "industry": industries,
    "size": sizes,
    "competitor": competitor,
    "months_since_loss": months_since_loss,
    "prev_contract_value": prev_contract_value,
    "last_contact_days": last_contact_days,
    "lost_reason_text": lost_reason_text,
    "team": teams,
    "winback": y
})
print(df.head())

# ============================================
# 2) 前処理
# ============================================
# カテゴリをLabel Encoding
for col in ["area","industry","size","competitor","team"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# テキストをTF-IDF化（日本語charベース）
vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
X_text = vec.fit_transform(df["lost_reason_text"])

# TF-IDFをDataFrame化して数値特徴と結合
tfidf_df = pd.DataFrame(X_text.toarray(), columns=vec.get_feature_names_out())
X = pd.concat([df.drop(columns=["lost_reason_text","winback"]), tfidf_df], axis=1)
y = df["winback"]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================
# 3) LightGBM 学習（callback形式 early_stopping）
# ============================================
train_data = lgb.Dataset(train_x, label=train_y)
test_data = lgb.Dataset(test_x, label=test_y, reference=train_data)

params = {
    "objective": "binary",
    "metric": ["auc"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

# callbackでearly_stoppingとログ出力を設定
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=["train","valid"],
    num_boost_round=300,
    callbacks=[early_stopping(20), log_evaluation(50)]
)

# ============================================
# 4) 評価
# ============================================
pred_proba = model.predict(test_x)
auc = roc_auc_score(test_y, pred_proba)
print(f"\nAUC: {auc:.3f}")
print(classification_report(test_y, (pred_proba>0.5).astype(int)))

# ============================================
# 5) チームごとの「取りやすさランク」
# ============================================
df["pred_proba"] = model.predict(X)
rank = (
    df.groupby("team")["pred_proba"]
    .apply(lambda x: x.sort_values(ascending=False).head(100).sum())
    .reset_index(name="expected_winbacks")
    .sort_values("expected_winbacks", ascending=False)
)
print("\n[チーム別 期待奪回数ランキング]")
print(rank)

# ============================================
# 6) 奪回確率上位企業リスト（チーム別）
# ============================================
top_targets = (
    df.sort_values("pred_proba", ascending=False)
      .groupby("team")
      .head(5)
      .sort_values(["team","pred_proba"], ascending=[True,False])
)
print("\n[チーム別 上位ターゲット例]")
print(top_targets[["team","area","industry","prev_contract_value","lost_reason_text","pred_proba"]].head(15))
