#ステップ2：性別ごとの満足度平均を算出

# 性別ごとの満足度平均を計算
gen_satisfaction = df.groupby('性別')['満足度'].mean().reset_index()

print("\n性別ごとの満足度平均:")
display(gen_satisfaction)

# グラフの作成
plt.figure(figsize=(7, 5))
plt.bar(gen_satisfaction['性別'], gen_satisfaction['満足度'], color=['skyblue', 'lightcoral'])
plt.xlabel('性別')
plt.ylabel('満足度平均')
plt.title('性別ごとの満足度平均')
plt.ylim(0, 5) # 満足度が1〜5なのでY軸の範囲を固定
plt.grid(axis='y', linestyle='--')
plt.show()
