# LightGBMモデルが「どの特徴量をどう使って予測しているか」をSHAP（特徴量の貢献度）で可視化


# ============================================
# Step 7：SHAP解析（LightGBMで要因を可視化）
# ============================================

!pip install shap -q

import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import numpy as np
import pandas as pd

# --- フォント設定（再利用） ---
font_candidates = [f for f in fm.findSystemFonts() if "ipag" in f.lower()]
font_path = font_candidates[0]
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False

print("✅ Step7：SHAP解析開始")

# ============================================
# ① SHAP値の計算（LightGBMモデルの説明）
# ============================================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_valid)

# ============================================
# ② 特徴量ごとの平均影響度（棒グラフ）
# ============================================
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
plt.title("特徴量の平均的影響度（SHAP値）", fontproperties=prop)
plt.tight_layout()
plt.show()

# ============================================
# ③ 各サンプルにおける影響方向（散布プロット）
# ============================================
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_valid, show=False)
plt.title("各特徴量の影響方向と分布（赤=使用量増、青=減）", fontproperties=prop)
plt.tight_layout()
plt.show()

# ============================================
# ④ 特定特徴量の依存関係（例：気温）
# ============================================
plt.figure(figsize=(7,5))
shap.dependence_plot("temperature", shap_values, X_valid, show=False)
plt.title("気温がガス使用量に与える影響（SHAP）", fontproperties=prop)
plt.tight_layout()
plt.show()

'''
横軸（x軸）→ 気温の値（例：0℃〜25℃）
縦軸（y軸）→ SHAP値（＝気温が予測ガス使用量をどの方向に動かしたか）
プラス（上側） → ガス使用量を増やす方向に働いた
マイナス（下側） → ガス使用量を減らす方向に働いた

'''
