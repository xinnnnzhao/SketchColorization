import torch
from torch import optim, nn
import numpy as np
from source.data.datas import read_img
from source.model.discriminator import Discriminator
from source.model.generator import Generator
from source.util.imgs import imageblur
import cv2


def show_img(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
batch_size = 2
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
    if e % 1 == 0:
        D.load()
        G.load()
    print(e)
    for i in range(length // batch_size):
        X_o_batch = X_o[i * batch_size:(i + 1) * batch_size]
        X_l_batch = X_l[i * batch_size:(i + 1) * batch_size]
        X_c_batch = np.array([imageblur(img) for img in X_o_batch])

        X_o_batch = 2.0 * (torch.from_numpy(X_o_batch).float().to(device=device) / 255.0) - 1.0
        X_l_batch = 2.0 * (torch.from_numpy(X_l_batch).float().to(device=device) / 255.0) - 1.0
        X_c_batch = 2.0 * (torch.from_numpy(X_c_batch).float().to(device=device) / 255.0) - 1.0
        pre_image = torch.cat([X_l_batch, X_c_batch], 3)  # (512,512,4)

        # Training discriminator
        D.zero_grad()
        # Real data
        d_real_data = torch.cat([pre_image, X_o_batch], 3)
        d_real_decision = D(d_real_data.view(batch_size, 7, 512, 512))
        d_real_error = criteria(d_real_decision, torch.ones(batch_size).to(device=device))
        # Fake data
        gen_img = G(pre_image.view(batch_size, 4, 512, 512))
        d_fake_data = torch.cat([pre_image, gen_img.view(batch_size, 512, 512, 3)], 3)
        d_fake_decision = D(d_fake_data.view(batch_size, 7, 512, 512))
        d_fake_error = criteria(d_fake_decision, torch.zeros(batch_size).to(device=device))

        d_error = d_fake_error + d_real_error
        d_error.backward()
        d_optimizer.step()

        # Training generator
        G.zero_grad()

        gen_img = G(pre_image.view(batch_size, 4, 512, 512))
        d_fake_data = torch.cat([pre_image, gen_img.view(batch_size, 512, 512, 3)], 3)
        dg_fake_decision = D(d_fake_data.view(batch_size, 7, 512, 512))
        g_error = criteria(dg_fake_decision, torch.ones(batch_size).to(device=device))
        g_error = g_error + torch.mean(torch.abs(gen_img.view(batch_size, 512, 512, 3) - X_o_batch))
        g_error.backward()
        g_optimizer.step()
        print("g_error: ", g_error.data, "d_error: ", d_error.data)


        # X_o_sample = X_o[0]
        # X_l_sample = X_l[0]
        # X_c_sample = np.array(imageblur(X_o_sample))
        # X_o_sample = 2.0 * (torch.from_numpy(X_o_sample).float().to(device=device)/255.0) - 1.0
        # X_l_sample = 2.0 * (torch.from_numpy(X_l_sample).float().to(device=device)/255.0) - 1.0
        # X_c_sample = 2.0 * (torch.from_numpy(X_c_sample).float().to(device=device)/255.0) - 1.0
        # pre_image = torch.cat([X_l_sample, X_c_sample], 2)
        # gen_img = G(pre_image.view(1, 4, 512, 512))
        # gen_img = ((gen_img.view(512,512,3) + 1.0) / 2.0) * 255.0
        # img = gen_img.cpu().data.numpy()
        # show_img(img)
