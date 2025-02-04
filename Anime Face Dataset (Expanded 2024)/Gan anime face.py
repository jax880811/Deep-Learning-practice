import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# 資料參數
IMAGE_DIR = r"E:\vscode\deep learning\Anime Face Dataset (Expanded 2024)\faces"
IMAGE_SHAPE = (96, 96, 3)  # 圖片大小
LATENT_DIM = 100  # 噪聲向量的維度
MODEL_NAME = "GAN_anime_face_model"  # 模型名稱
OUTPUT_DIR = r"E:\vscode\deep learning\Anime Face Dataset (Expanded 2024)\圖片觀察"  # 圖片觀察儲存位置

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定義生成器
def build_generator():
    model = Sequential()
    model.add(Dense(12 * 12 * 256, activation="relu", input_dim=LATENT_DIM))  # 初始大小調整為 12x12
    model.add(Reshape((12, 12, 256)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))  # 12x12 -> 24x24
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"))  # 24x24 -> 48x48
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh"))  # 48x48 -> 96x96
    return model

# 定義判別器
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=IMAGE_SHAPE))  # 96x96 -> 48x48
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))  # 48x48 -> 24x24
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))  # 24x24 -> 12x12
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model

# 定義 GAN 模型
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(LATENT_DIM,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    return Model(gan_input, gan_output)

# 加載資料
def load_images(image_dir, image_shape):
    images = []
    for file_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file_name)
        img = load_img(img_path, target_size=image_shape[:2])  # 圖片大小保持為 96x96
        img_array = img_to_array(img) / 127.5 - 1.0  # 正規化到 [-1, 1]
        images.append(img_array)
    return np.array(images)

# 保存生成圖片
def save_generated_images(generator, epoch, output_dir):
    noise = np.random.normal(0, 1, (36, LATENT_DIM))  # 隨機生成 36 張圖片
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) * 127.5  # 轉換到 [0, 255]
    generated_images = generated_images.astype(np.uint8)

    # 繪製 6x6 圖片方格
    fig, axs = plt.subplots(6, 6, figsize=(12, 12))
    cnt = 0
    for i in range(6):
        for j in range(6):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"epoch_{epoch}.png")
    plt.savefig(output_path)
    plt.close()

# 訓練模型
def train_gan(epochs, batch_size, save_path):
    # 建立模型
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

    # 加載資料
    dataset = load_images(IMAGE_DIR, IMAGE_SHAPE)
    half_batch = batch_size // 2

    for epoch in range(1, epochs + 1):
        # 訓練判別器
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        real_images = dataset[idx]
        noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
        fake_images = generator.predict(noise)

        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 訓練生成器
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # 每隔 100 個 epoch 打印損失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        # 每隔 100 個 epoch 保存生成圖片
        if epoch % 100 == 0:
            save_generated_images(generator, epoch, OUTPUT_DIR)
            generator.save(save_path)

# 執行訓練
train_gan(epochs=5000, batch_size=64, save_path=f"{MODEL_NAME}.h5")