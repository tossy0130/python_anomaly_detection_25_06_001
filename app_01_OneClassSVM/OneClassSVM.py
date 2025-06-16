import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler # 標準化用

### 画像を扱う
import cv2
import glob

# 画像読み込み
train_images = glob.glob("./data/*")

# リストに格納
train_data = []
for i in train_images:
  image = cv2.imread(i) # 1枚ずつ読み込み
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR から、 RGB へ変換
  train_data.append(image)

# array 型に変換
train_data = np.array(train_data)
# print(train_data)


# 出力 : (38, 128, 128, 3)
train_data.shape

# 1本の長い、ベクトルへ変換
train_data = train_data.reshape(train_data.shape[0], -1)
train_data.shape

### データの正規化
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# print(train_data[1, :10])

### OneClassSVM 作成
model = OneClassSVM()
model.fit(train_data)

########################
###### テストデータ作成
########################

# 画像読み込み
test_images = glob.glob("./test_data/*")

# リストに格納
test_data_original = []

for i in test_images:
  image = cv2.imread(i) # 1枚ずつ読み込み
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR から、 RGB へ変換
  test_data_original.append(image)

# array 型に変換
test_data = np.array(test_data_original)
# print(test_data)

# 出力 : (38, 128, 128, 3)
test_data.shape

# 1本の長い、ベクトルへ変換
test_data = test_data.reshape(test_data.shape[0], -1)
test_data.shape

### データの正規化
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)

#########
###### モデルのスコアを表示
#########
prediction_score = model.decision_function(test_data)
print("テストデータスコア:::\n" , prediction_score)

###### テストデータオリジナル　表示 （画像も表示）
plt.figure(figsize=(10, 10))
for i in range(len(test_data_original)):
  plt.subplot(3, 4, i+1)
  plt.imshow(test_data_original[i])
  plt.title(f"image {i + 1} \n,{prediction_score[i]:.2f}")

plt.tight_layout
plt.show()