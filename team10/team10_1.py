# ============================================
# Step 0：日本語フォント設定（Colab文字化け防止）
# ============================================
!apt-get -y install fonts-ipafont-gothic > /dev/null

import subprocess
subprocess.run(["fc-cache", "-fv"], check=True)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

# --- フォント検出＆登録 ---
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
if not font_candidates:
    raise FileNotFoundError("IPAフォントが見つかりません。ランタイムを再起動して再実行してください。")
font_path = font_candidates[0]
fm.fontManager.addfont(font_path)
mpl.rc("font", family="IPAGothic")
mpl.rcParams["axes.unicode_minus"] = False
print(f"✅ 日本語フォント設定完了：{os.path.basename(font_path)}")

# ============================================
# Step 1〜3：前処理とモデル準備
# ============================================
!pip install -q sentence-transformers scikit-learn pandas numpy matplotlib

import pandas as pd, numpy as np, re, random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ==== サンプル営業メモ ====
salespersons = ["Aさん","Bさん","Cさん","Dさん","Eさん"]
notes = [
    "料金プランの違いを説明し、省エネ施策を提案。電力量データの可視化に関心あり。",
    "他社比較を求められた。コスト削減効果の試算を提示し、来月見積提出予定。",
    "トラブル対応の不満があったため、保守サポートの強化策を説明。謝罪と次回フォローを約束。",
    "再エネメニューの相談。環境価値と長期的な単価リスクを説明。意思決定者同席を依頼。",
    "基本料金の見直し要望。需要家側のピークカット提案、BEMS連携の事例を紹介。",
    "省エネ補助金の情報提供。導入ステップとスケジュールを共有して合意形成を図った。",
    "契約更新に向けたクロージング。年間削減見込みを再提示し、社内稟議の進め方を助言。",
    "苦情対応：請求金額の誤認が発覚。明細の内訳を説明し、次回までに再発防止策を提出。",
    "需要予測の話題から、デマンドレスポンスの可能性を提示。小規模からのPoC提案。",
    "意思決定者が不在。傾聴に徹し、現状課題を整理。次回アジェンダを合意。"
]
data = [{"salesperson": random.choice(salespersons), "note": random.choice(notes)} for _ in range(120)]
df = pd.DataFrame(data)
df["text"] = df["note"].apply(lambda s: re.sub(r"\s+", " ", s).strip())

# ==== SentenceTransformer ====
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
embeddings = normalize(embeddings)

# ============================================
# Step 4：クラスタ数自動最適化＋クラスタリング
# ============================================
scores = []
K_range = range(4, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    scores.append(score)
best_k = K_range[np.argmax(scores)]
print(f"最適クラスタ数: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(embeddings)

# ============================================
# Step 5：各クラスタ代表文
# ============================================
rep_texts = []
for c in range(best_k):
    idx = np.where(df["cluster"] == c)[0]
    cluster_vecs = embeddings[idx]
    centroid = cluster_vecs.mean(axis=0)
    sims = cosine_similarity(cluster_vecs, [centroid])
    rep_text = df.iloc[idx[np.argmax(sims)]]["text"]
    rep_texts.append(rep_text)
    print(f"=== Cluster {c} の代表文 ===")
    print(rep_text)

# ============================================
# Step 6：AIスキル命名（Sentence類似度）
# ============================================
skill_def = {
    "提案力": "顧客の課題を理解し最適な提案を行うスキル。",
    "課題解決力": "課題を発見し、解決策を導き出すスキル。",
    "傾聴力": "顧客の話を丁寧に聞き、要望を引き出すスキル。",
    "説明力": "専門的な内容をわかりやすく説明するスキル。",
    "クレーム対応": "苦情に冷静に対応し、信頼回復を行うスキル。",
    "交渉力": "条件交渉を円滑に進めるスキル。",
    "クロージング": "商談をまとめ契約に導くスキル。",
    "関係構築力": "長期的な信頼関係を築くスキル。",
    "チーム連携力": "社内外で協力し課題を解決するスキル。",
    "顧客理解力": "顧客の業種や背景を把握し、最適な対応を行うスキル。",
}
skill_keys = list(skill_def.keys())
skill_texts = list(skill_def.values())

cluster_emb = model.encode(rep_texts)
skill_emb = model.encode(skill_texts)
sim = cosine_similarity(cluster_emb, skill_emb)

auto_skill_names, auto_scores = [], []
for i in range(best_k):
    idx = np.argmax(sim[i])
    score = sim[i, idx]
    name = skill_keys[idx] if score >= 0.35 else "その他"
    auto_skill_names.append(name)
    auto_scores.append(score)

# 重複整理
used = set()
for i in range(best_k):
    if auto_skill_names[i] in used and auto_skill_names[i] != "その他":
        auto_skill_names[i] = "その他"
    else:
        used.add(auto_skill_names[i])

print("\n=== 🧠 自動スキルカテゴリ名 ===")
for i, (name, sc) in enumerate(zip(auto_skill_names, auto_scores)):
    print(f"Cluster {i} → {name}（類似度: {sc:.2f}）")

df["cluster_name"] = df["cluster"].apply(lambda x: auto_skill_names[int(x)])

# ============================================
# Step 7：その他再分類＋再グラフ
# ============================================
other_texts = df[df["cluster_name"] == "その他"]["text"].tolist()
if len(other_texts) > 10:
    print("\n=== その他クラスタ再分析 ===")
    other_emb = model.encode(other_texts)
    sub_k = min(3, len(other_texts))
    sub_kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init="auto")
    sub_labels = sub_kmeans.fit_predict(other_emb)

    refined_labels, sub_rep_texts = [], []
    for i in range(sub_k):
        idx = np.where(sub_labels == i)[0]
        centroid = other_emb[idx].mean(axis=0)
        sims = cosine_similarity(other_emb[idx], [centroid])
        rep_text = other_texts[idx[np.argmax(sims)]]
        sub_rep_texts.append(rep_text)
        print(f"\n--- サブクラスタ{i} 代表文 ---\n{rep_text}")

    sub_rep_emb = model.encode(sub_rep_texts)
    sim_sub = cosine_similarity(sub_rep_emb, skill_emb)

    for i, t in enumerate(sub_rep_texts):
        best_skill_idx = np.argmax(sim_sub[i])
        score = sim_sub[i, best_skill_idx]
        if score > 0.4:
            refined_labels.append(skill_keys[best_skill_idx])
        else:
            refined_labels.append(f"{t[:5]}対応力")

    # 再分類を反映（df本体に上書き）
    df_other = df[df["cluster_name"] == "その他"].copy()
    df_other_emb = model.encode(df_other["text"].tolist())
    df_other["sub_cluster"] = sub_kmeans.fit_predict(df_other_emb)
    df_other["cluster_name"] = df_other["sub_cluster"].apply(lambda x: refined_labels[x])

    df = pd.concat([
        df[df["cluster_name"] != "その他"],
        df_other
    ], ignore_index=True)

    print("✅ その他を再分類し、dfに統合しました。")
else:
    print("\n⚪ その他が少ないため再クラスタリングをスキップしました。")

# ============================================
# Step 8：最終グラフ出力（再分類後）
# ============================================
pivot_final = df.groupby(["salesperson", "cluster_name"]).size().unstack(fill_value=0)
pivot_final.plot(kind="bar", figsize=(10,6))
plt.title("担当者 × 最終スキルカテゴリ（再分類後）", fontsize=14)
plt.xlabel("担当者", fontsize=12)
plt.ylabel("件数", fontsize=12)
plt.legend(title="スキルカテゴリ", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
