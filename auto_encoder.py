import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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

##### 0 ～ 1　の範囲にする。（255 で割る）
train_data = np.array(train_data)

# float 型に変換
train_data = train_data.astype(float) / 255

###############
###### ハイパーパラメータ
###############
LEARNING_RATE = 0.001
BATCH_SIZE = 8
EPOCHS = 50

###############
###### エンコーダ
###############

# 128 × 128 のピクセルが ３つ
encoder_input = Input(shape=(128, 128, 3))

# padding='same' => 画像サイズが変わらないように、 0 パディング
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

###############
###### デコーダ
###############
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

######### シグモイド
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

######### モデル
autoencoder = Model(encoder_input, decoded)
autoencoder.summary()

###############
###### オプティマイザーの設定
###############
optimizer = Adam(learning_rate=LEARNING_RATE)

# loss ロス => mse , メトリックス => mae , アキュラシー（accuracy）
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mae", "accuracy"])

# ============================
# *** 学習 の 実行 ***
# ============================
autoencoder.fit(train_data, train_data, batch_size=BATCH_SIZE, epochs=EPOCHS)