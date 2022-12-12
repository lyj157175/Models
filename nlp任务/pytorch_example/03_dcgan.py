import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(noise_size, g_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_size * 8, g_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_size * 4, g_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_size * 2, g_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_size, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # input: 64, 512, 1, 1
        output = self.conv(input)
        return output   # 64, 3, 64, 64



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, d_size, 4, 2, 1, bias=False),    # 64, 64, 32, 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_size, d_size * 2, 4, 2, 1, bias=False),    # 64, 128, 16, 16
            nn.BatchNorm2d(d_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_size * 2, d_size * 4, 4, 2, 1, bias=False),  # 64, 256, 8, 8
            nn.BatchNorm2d(d_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_size * 4, d_size * 8, 4, 2, 1, bias=False),  # 64, 512, 4, 4
            nn.BatchNorm2d(d_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_size * 8, 1, 4, 1, 0, bias=False),  # 64, 1, 1, 1
            nn.Sigmoid()
        )

    def forward(self, input):
        # input: 64, 3, 64, 64
        output = self.conv(input)  # 64, 1, 1, 1
        return output.view(-1, 1).squeeze(1)  # 64




if __name__ == '__main__':
    # dataset = 'cifar10'   # 'cifar10 | lsun | mnist |imagenet | folder | lfw | fake'
    random.seed(123)
    torch.manual_seed(123)
    cudnn.ben_channelshmark = True
    n_channels = 3
    ngpu = 1
    noise_size = 100
    g_size = 64  # 生成器特征维度
    d_size = 64  # 判别器特征维度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = 'ouput'
    #
    dataset = dset.CIFAR10(root='data', download=False,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=2)

    # build model
    model_g = Generator(ngpu).to(device)
    model_g.apply(weights_init)
    model_d = Discriminator(ngpu).to(device)
    model_d.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(model_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(model_g.parameters(), lr=0.0002, betas=(0.5, 0.999))


    epochs = 25
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            # data[0] : 64, 3, 64, 64
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            model_d.zero_grad()
            real_cpu = data[0].to(device)   # 64, 3, 64, 64 
            batch_size = real_cpu.size(0)  # 64
            label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)  # (64, 1)
            output = model_d(real_cpu)   # 64
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)  # 64, 100, 1, 1
            fake = model_g(noise)  # 64, 3, 64, 64
            label.fill_(fake_label)
            output = model_d(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = model_d(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu, '%s/real_samples.png' % out_path, normalize=True)
                fake = model_g(fixed_noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (out_path, epoch), normalize=True)

        # do checkpointing
        torch.save(model_g.state_dict(), '%s/model_g_epoch_%d.pth' % (out_path, epoch))
        torch.save(model_d.state_dict(), '%s/model_d_epoch_%d.pth' % (out_path, epoch))
