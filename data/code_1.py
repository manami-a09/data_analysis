#前準備：必要なライブラリのインストールとインポート

# japanize_matplotlib がインストールされていない場合、以下の行を実行してインストールします。
# 初回のみ実行すればOKです。
!pip install -q japanize-matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib # これをインポートすることでmatplotlibで日本語が使えるようになります
import seaborn as sns # グラフをよりきれいに描画するためのライブラリをインポートします

print("必要なライブラリのインポートが完了しました。")
