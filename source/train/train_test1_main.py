import torch
from torch import optim, nn
import numpy as np
from source.data.datas import read_img
from source.model.discriminator import Discriminator
from source.model.generator import Generator
from source.util.imgs import imageblur

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
batch_size = 10
g_steps = 1
print_interval = 10

G = Generator().to(device)
D = Discriminator().to(device)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
criteria = nn.BCELoss()

X_o, X_l = read_img(20)
length = X_o.shape[0]
for e in range(1000):
    print(e)
    for i in range(length // batch_size):
        X_o_batch = 2 * (X_o[i * batch_size:(i + 1) * batch_size] / 255) - 1
        X_l_batch = 2 * (X_l[i * batch_size:(i + 1) * batch_size] / 255) - 1
        X_c_batch = 2 * (np.array([imageblur(img) for img in X_o_batch]) / 255) - 1

        X_o_batch = torch.from_numpy(X_o_batch).float().to(device=device)
        X_l_batch = torch.from_numpy(X_l_batch).float().to(device=device)
        X_c_batch = torch.from_numpy(X_c_batch).float().to(device=device)
        pre_image = torch.cat([X_l_batch, X_c_batch], 3)  # (512,512,4)

        # Training discriminator
        D.zero_grad()
        # Real data
        d_real_data = torch.cat([pre_image, X_o_batch], 3)
        print(d_real_data.shape)
        d_real_decision = D(d_real_data)
        d_real_error = criteria(d_real_decision, torch.ones(batch_size))
        # Fake data
        gen_img = G(pre_image)
        d_fake_data = torch.cat([pre_image, gen_img], 3)
        d_fake_decision = D(d_fake_data)
        d_fake_error = criteria(d_fake_decision, torch.zeros(batch_size))

        d_error = d_fake_error + d_real_error
        d_error.backward()
        d_optimizer.step()

        g_error = 0
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            g_fake_data = G(pre_image)
            dg_fake_decision = D(g_fake_data)
            g_error = criteria(dg_fake_decision,
                               torch.ones(batch_size))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        if e % print_interval == 0:
            print("epoch: %d, i : %d,  D-loss: %d, G-loss: %d" % (
                e, i, d_error.detach().numpy(), g_error.detach().numpy()))
