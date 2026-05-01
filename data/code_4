#ステップ3：年代ごとの回答数を集計

# 年代ごとの回答数を集計
age_counts = df.groupby('年代').size().reset_index(name='回答数')

# 年代をソートする（必要であれば）
age_counts['年代'] = pd.Categorical(age_counts['年代'], categories=sorted(age_counts['年代'].unique()), ordered=True)
age_counts = age_counts.sort_values('年代')

print("\n年代ごとの回答数:")
display(age_counts)

# グラフの作成
plt.figure(figsize=(10, 6))
sns.barplot(x='年代', y='回答数', data=age_counts, palette='viridis')
plt.xlabel('年代')
plt.ylabel('回答数')
plt.title('年代ごとの回答数')
plt.grid(axis='y', linestyle='--')
plt.show()
