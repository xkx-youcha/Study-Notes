# 2024/9/25
""" 基于MNIST 实现生成对抗网络（GAN） """

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

image_size = [1, 28, 28]
device = torch.cuda.is_available()

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, Z):
        # shape of Z: [batchsize, latent_dim]
        output = self.model(Z)
        image = output.reshape(Z.shape[0], *image_size)
        return image

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, image):
        # shape of image: [batchsize, 1, 28, 28]
        # 将四维转换为二维传入model
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob

# 获取MNIST数据集
dataset = torchvision.datasets.MNIST("../data", train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(28),
                                         torchvision.transforms.ToTensor(),
                                         # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                                     ]))
# print(len(dataset))     # 60000
# print(dataset[0][0].shape)      # torch.Size([1, 28, 28])

batch_size = 64
latent_dim = 96
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 实例化判别器和生成器
generator = Generator()
discriminator = Discriminator()

# 构造优化器
# 需要两个优化器，分别对判别器和生成器进行优化
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.003, betas=(0.4, 0.8), weight_decay=0.0001 )
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.003, betas=(0.4, 0.8), weight_decay=0.0001)

# 损失函数，因为最后是进行二分类判别
loss_fn = nn.BCELoss()

label_one = torch.ones(batch_size, 1)
label_zero = torch.zeros(batch_size, 1)

if device:
    print("use cuda for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    label_one = label_one.cuda()
    label_zero = label_zero.cuda()

# 开始训练
epochs = 100
for epoch in range(epochs):
    for index, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch

        # 随机生成z，z是服从正态分布的，形状是batch_size * 1 * 28 * 28
        z = torch.randn(batch_size, latent_dim)
        if device:
            gt_images = gt_images.to("cuda")
            z = z.to("cuda")
        # print(z.shape)
        pred_images = generator(z)
        # print(pred_images.shape)

        # 生成器的部分
        g_optimizer.zero_grad()
        g_loss = loss_fn(discriminator(pred_images), label_one)
        g_loss.backward()
        g_optimizer.step()

        # 判别器的部分
        d_optimizer.zero_grad()
        # 判别器希望真实的图片全部判断为1，生成的图片全部判断为0
        # 训练过程中，观察d_real_loss和d_fake_loss，是否同时下降并达到最小值（并且差不多大），则说明稳定了
        d_real_loss = loss_fn(discriminator(gt_images), label_one)
        d_fake_loss = loss_fn(discriminator(pred_images.detach()), label_zero)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        d_loss.backward()
        d_optimizer.step()

        # if index % 1000 == 0:
        #     for i, image in enumerate(pred_images):
        #         torchvision.utils.save_image(image, f"./image/image_{i}.png")

        if index % 50 == 0:
            print(f"step:{len(dataloader)*epoch+index}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, "
                  f"real_loss:{d_real_loss.item()}, fake_loss:{d_fake_loss.item()}")

        if index % 400 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image, f"./fake_image/image_{len(dataloader)*epoch+index}.png", nrow=4)