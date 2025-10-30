# ページタイトル、検索語をトッピック分類し、失注率を算出 　※コード1実行後


# ============================================
# Step 5：ページ内容の自動トピック分類（NMF分析）
# ============================================
!pip -q install scikit-learn

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# === 1. データを結合（GA4 × CRM） ======================
data = sessions.merge(crm, on="user_pseudo_id", how="left")

# === 2. テキストデータの準備 ============================
# ページタイトルと検索語を結合して、文章として扱う
data["text"] = (data["page_title"].fillna("") + " " + data["search_term"].fillna("")).str.strip()

# 空文字を除外
data = data[data["text"].str.len() > 0]

# === 3. TF-IDFベクトル化 ===============================
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(data["text"])

# === 4. NMF（非負値行列因子分解）でトピック抽出 ========
nmf = NMF(n_components=6, random_state=42, init="nndsvd", max_iter=400)
W = nmf.fit_transform(X_tfidf)  # 各セッションのトピック強度
H = nmf.components_              # 各トピックの単語構成

# === 5. 各トピックの代表単語を抽出 =====================
terms = np.array(tfidf.get_feature_names_out())
topic_keywords = []
for k in range(nmf.n_components):
    top_idx = H[k].argsort()[-6:][::-1]
    words = ", ".join(terms[top_idx])
    topic_keywords.append(words)

# === 6. 各セッションの最も強いトピックを決定 ============
topic_cols = [f"topic_{i}" for i in range(nmf.n_components)]
topic_strength = pd.DataFrame(W, columns=topic_cols)
data = data.reset_index(drop=True).join(topic_strength)
data["main_topic"] = topic_strength.idxmax(axis=1)

# === 7. トピック別の失注率を計算 ========================
topic_loss_rate = data.groupby("main_topic")["lost_flag"].mean().sort_values(ascending=False)

print("【失注率が高いトピック順】\n")
for topic_name, loss_rate in topic_loss_rate.items():
    idx = int(topic_name.split("_")[-1])
    print(f"{topic_name}: 失注率={loss_rate:.2f}  代表語: {topic_keywords[idx]}")

# === 8. グラフで可視化（日本語フォント対応） ============
plt.figure(figsize=(8,4))
topic_loss_rate.plot(kind="bar", color="salmon")
plt.title("トピック別の失注率", fontproperties=prop)
plt.ylabel("失注率", fontproperties=prop)
plt.xlabel("トピック", fontproperties=prop)
plt.xticks(rotation=45, fontproperties=prop)
plt.tight_layout()
plt.show()
