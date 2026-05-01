#ステップ1：Excelファイルの読み込み

# ExcelファイルをDataFrameとして読み込む
# 'engine='openpyxl'' を指定することで、より安定してExcelファイルを読み込めます。
#ファイルのパスはcode_0で確認
file_path = '/content/sample.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"'{file_path}' の読み込みに成功しました。")
    # 読み込んだデータの最初の5行を表示して内容を確認
    print("\nデータフレームの最初の5行:")
    display(df.head())
    print("\nデータフレームの情報 (列名、データ型、欠損値など):")
    df.info()
except FileNotFoundError:
    print(f"エラー: '{file_path}' が見つかりませんでした。ファイルパスを確認してください。")
except Exception as e:
    print(f"ファイル読み込み中にエラーが発生しました: {e}")
