#colabにファイルをアップロードした時にデフォルトで/content に保存される
#このコードでファイルのパスを確認

import os

# List files in the current working directory, which is usually /content/ in Colab
current_directory = '/content/'
print(f"Files in '{current_directory}':")
for item in os.listdir(current_directory):
    print(os.path.join(current_directory, item))


  
