# =====================================================
# 🚀 Colab最終完成版：意味・期間・前後文を考慮した日本語検索ツール
# =====================================================
!pip install rank-bm25 fugashi ipadic scikit-learn tqdm ipywidgets pandas sentence-transformers --quiet

import re
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import fugashi, ipadic
import ipywidgets as widgets
from IPython.display import display
from sentence_transformers import SentenceTransformer, util

# --- 初期化 ---
tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)
embedder = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
print("✅ MeCab初期化完了（UTF-8辞書）")
print("✅ Sentence-BERT読込完了（日本語対応）")

# =====================================================
# Step 1: サンプル文書生成
# =====================================================
def make_sample_docs():
    docs = {
        "営業施策_省エネ補助.txt": (
            "2025年度 省エネ補助金の申請スケジュール（想定）。\n"
            "一次公募：2月上旬～3月中旬。\n"
            "二次公募：5月～6月。\n"
            "上限1,500万円、補助率1/2。\n"
            "対象：高効率空調、インバータ、LED更新、BEMS等。\n"
            "営業は高圧需要家の冷凍冷蔵・空調更新の掘り起こしを優先。\n"
            "省エネ診断の無料クーポンを配布。\n"
        ),
        "料金改定_2025Q1.txt": (
            "2025年1月実施の料金改定に関する社内向け周知。\n"
            "需要家への影響は平均+3.2%。低圧は+2.0%、高圧は+3.8%、特別高圧は+4.1%。\n"
            "燃料費調整単価の見直しと再エネ賦課金の増額が主因。\n"
            "営業部は中小企業向けの説明資料（Q&A）を1月10日までに提出。\n"
        ),
        "QandA_料金改定.txt": (
            "Q: 料金改定の理由は？\n"
            "A: 国際エネルギー価格の高止まり、為替、再エネ関連費用の増加が要因です。\n"
            "Q: 値上がり幅は？\n"
            "A: 低圧+2.0%、高圧+3.8%、特別高圧+4.1%の見込みです。\n"
            "Q: 影響を抑える方法は？\n"
            "A: 使用時間帯のシフト、基本料金の見直し、省エネ設備更新の検討が有効です。\n"
        )
    }
    for name, text in docs.items():
        (DATA_DIR / name).write_text(text, encoding="utf-8")
    print("✅ サンプル文書を data/ に作成しました")

# =====================================================
# Step 2: 文分割＆形態素解析
# =====================================================
def read_docs() -> List[Dict[str, Any]]:
    items = []
    for p in DATA_DIR.glob("*.txt"):
        raw = p.read_text(encoding="utf-8")
        raw = raw.replace("\r\n", "。").replace("\n", "。")
        sents = re.split(r"(?<=[。．!！?？])", raw)
        for i, s in enumerate(sents):
            s = s.strip()
            if len(s) > 1:
                items.append({"doc": p.name, "sent_id": i, "text": s})
    return items

def tokenize_ja(text):
    return [word.surface for word in tagger(text)]

# =====================================================
# Step 3: 検索モデル構築（TF-IDF + BM25）
# =====================================================
def build_models(texts: List[str]):
    tokenized_texts = [" ".join(tokenize_ja(t)) for t in texts]
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(tokenized_texts)
    bm25 = BM25Okapi([t.split() for t in tokenized_texts])
    return vec, X, bm25

# =====================================================
# Step 4: 検索（意味＋期間＋前後文補強）
# =====================================================
def search_with_context(query: str, rows: List[Dict[str, Any]], vec, X, bm25, topk=5, min_match=2):
    query_tok = tokenize_ja(query)
    qv = vec.transform([" ".join(query_tok)])
    sim = cosine_similarity(qv, X).ravel()
    bm25_scores = bm25.get_scores(query_tok)

    fused = Counter()
    for i in range(len(rows)):
        fused[i] = sim[i] + bm25_scores[i]

    prelim = [i for i, _ in fused.most_common(topk*3)]
    results = []

    for i in prelim:
        text = rows[i]["text"]
        overlap = len(set(query_tok) & set(tokenize_ja(text)))
        if overlap < min_match:
            continue

        doc_name = rows[i]["doc"]
        doc_rows = [r["text"] for r in rows if r["doc"] == doc_name]

        # ✅ 改良版：関連文＋前後2文を結合
        related = []
        for j, t in enumerate(doc_rows):
            if any(w in t for w in query_tok):
                start = max(0, j - 2)
                end = min(len(doc_rows), j + 3)
                related.extend(doc_rows[start:end])
        context_text = " ".join(sorted(set(related)))

        results.append({
            "doc": doc_name,
            "text": context_text,
            "score": float(fused[i])
        })

    if not results:
        return []

    # --- 意味類似度を計算 ---
    texts = [r["text"] for r in results]
    q_emb = embedder.encode([query], convert_to_tensor=True)
    t_embs = embedder.encode(texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, t_embs)[0]
    for i, r in enumerate(results):
        r["semantic_score"] = float(cos_scores[i])
        r["final_score"] = 0.6 * r["score"] + 0.4 * r["semantic_score"]

        # ✅ 期間表現を含む文をスコア補正（＋0.5）
        if re.search(r"\d{1,2}月|上旬|中旬|下旬|年度|月", r["text"]):
            r["final_score"] += 0.5

    merged_results = {}
    for r in sorted(results, key=lambda x: -x["final_score"]):
        if r["doc"] not in merged_results:
            merged_results[r["doc"]] = r
        else:
            merged_results[r["doc"]]["text"] += " " + r["text"]

    return list(merged_results.values())[:topk]

# =====================================================
# Step 5: 要約（上位文の抜粋）
# =====================================================
def simple_summary(cands: List[Dict[str, Any]], max_sent=3):
    seen, picked = set(), []
    for c in cands:
        t = re.sub(r"\s+", "", c["text"])
        if t[:30] not in seen:
            picked.append(c["text"])
            seen.add(t[:30])
        if len(picked) >= max_sent:
            break
    return " / ".join(picked)

# =====================================================
# Step 6: UI（常に入力可能）
# =====================================================
output_area = widgets.Output()

def run_search(query):
    with output_area:
        output_area.clear_output(wait=True)
        print(f"🔍 検索クエリ: {query}\n")
        if not any(DATA_DIR.glob("*.txt")):
            make_sample_docs()

        rows = read_docs()
        texts = [r["text"] for r in rows]
        vec, X, bm25 = build_models(texts)
        hits = search_with_context(query, rows, vec, X, bm25, topk=5)
        if not hits:
            print("該当する文が見つかりませんでした。")
            return
        summary = simple_summary(hits, 3)

        df = pd.DataFrame(hits)[["doc", "text", "final_score"]]
        df["final_score"] = df["final_score"].round(3)
        display(df.style.set_properties(**{'white-space': 'pre-wrap'}))

        print("\n--- 抽出要約 ---")
        print(summary)
        print("\n💡 クエリを変更して再実行できます。")

query_box = widgets.Text(
    value='省エネ 補助金 スケジュール',
    placeholder='検索キーワードを入力',
    description='検索クエリ:',
    layout=widgets.Layout(width='70%')
)
run_button = widgets.Button(description="検索実行", button_style='success')

def on_click(b):
    run_search(query_box.value)

run_button.on_click(on_click)
display(widgets.VBox([widgets.HBox([query_box, run_button]), output_area]))
