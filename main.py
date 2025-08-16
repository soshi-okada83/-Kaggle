import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import tree

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
# print(df.isnull().sum())

# 乗船した港を確認する
# plt.figure(figsize=(10, 6))
# sns.countplot(x = 'Embarked', data=df)
# plt.show()

# 元データをコピー
df2 = df.copy()

# 欠損値の補完
df2.Embarked = df2.Embarked.fillna('S')
# print(df2.Embarked.isnull().sum())

# 年齢の最小値と最大値を確認
# print(df2.Age.max())
# print(df2.Age.min())

# 年齢を八個に分けてヒストグラムを作成する
# sns.displot(df.Age, bins=8, kde=False)
# plt.show()

# 年齢の平均値と中央値を確認する
# print(df2.Age.mean())
# print(df2.Age.median())

# df2をコピー
df3 = df2.copy()

# 年齢の中央値を計算
age_median = df3.Age.median()

# 年齢欠損値の補完
df3.Age = df3.Age.fillna(age_median)
# print(df3.Age.isnull().sum())

# 使わないカラムの削除
df4 = df3.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'])

# Embarkedの数値変換（ワンホットエンコーディング）
tmp_embarked = pd.get_dummies(df4['Embarked'], dtype=int, prefix='Embarked')

# 元データフレームにワンホット結果を連結して、変数df5に格納する
df5 = pd.concat([df4, tmp_embarked], axis=1).drop(columns=['Embarked'])

# Sexの数値変換（ラベルエンコーディング）
df5['Sex'] = pd.get_dummies(df5['Sex'], dtype=int, drop_first=True)

# --------学習データとテストデータに分割する-------

train = df5[~df5.Survived.isnull()] # 学習データに分割した結果を変数trainに格納
test = df5[df5.Survived.isnull()] # テストデータに分割した結果を変数testに格納

# テストデータからSurvivedを削除
test = test.drop(columns=['Survived'])

# 正解をy_trainに格納
y_train = train.Survived

# 特徴量をX_trainに格納
X_train = train.drop(columns=['Survived'])

# 決定木モデルの準備
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# 作成した決定木モデルを使った予測を行う
y_pred = model.predict(test)
# print(y_pred)

# テストデータと予測結果の大きさを確認する
# print(len(test), len(y_pred))

# 予測結果をテストデータに反映する
test['Survived'] = y_pred
# print(test.head())

# 提出用データマートを作成する
pred_df = test[['PassengerId', 'Survived']].set_index('PassengerId')

# 予測結果を整数に変換する
pred_df.Survived = pred_df.Survived.astype(int)

# CSVの作成
pred_df.to_csv('submission_v1.csv', index_label=['PassengerId'])