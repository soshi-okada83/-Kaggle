import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# データの読み込み
dir_path = 'titanic'
train_df = pd.read_csv(os.path.join(dir_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(dir_path, 'test.csv'))

# 学習データの先頭五行を見てみる
# print(train_df.head())

# テストデータの先頭五行を見てみる
# print(test_df.head())

# データフレームの大きさ
# print(train_df.shape)
# print(test_df.shape)

# データを見るために学習データとテストデータを連結
df = pd.concat([train_df, test_df], ignore_index=True)

# データ内の欠損値を確認する
print(df.isnull().sum())