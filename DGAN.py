from tqdm import tqdm
import numpy as np
import PIL.Image
import torch
import legacy
import itertools
import copy
from training.networks import Generator, SynthesisNetwork

#pts = np.array([[754, 409], [752, 623]])
#tar = np.array([[739, 396], [735, 627]]) 


c = 20
r1 = 3
r2 = 11
#n = pts.shape[0]
device = torch.device('cuda')
network_path = 'networks/ffhq.pkl'
z_path = "data/DGAN/ffhq/z-5.npz"
number = 5

print('Loading Network...')
with open(network_path, 'rb') as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

res = G.img_resolution    

def get_w(z_path):
    z = np.load(z_path)['z']
    w = G.mapping(torch.from_numpy(z).to(device), None)
    return w

def create_generator():
    gen = Generator(w_dim=G.w_dim, img_resolution=G.img_resolution, img_channels=G.img_channels, z_dim=G.z_dim, c_dim=G.c_dim)
    gen.load_state_dict(G.state_dict(), strict=False)
    return gen

gen = create_generator().to(device)
gen.requires_grad_(False)
gen.eval()

'''
Returns an array of points that are within distance r from point p; ord is 1 for sup and 2 for euclidean
distance
'''
def sigma_ball(p, r, ord):
    points = np.array([p])
    if ord == 2:
        for x in range(-r, r):
            for y in range(-r, r):
                if x**2 + y**2 <= r:
                    if (p[0]+x >= 0) and (p[1]+y >= 0) and (p[0]+x < res) and (p[1]+y < res):
                        points = np.append(points, [[p[0]+x, p[1]+y]], axis=0)
    if ord == 1:
        for x in range(-r, r):
            for y in range(-r, r):
                if (p[0]+x >= 0) and (p[1]+y >= 0) and (p[0]+x < res) and (p[1]+y < res):
                    points = np.append(points, [[p[0]+x, p[1]+y]], axis=0)
    return points[1:]

'''
q is a tuple (x, y) containing pixel location. we use bilinear interpolation to find the 
feature value after moving it along unit vector pointing towards corresponding target point
'''
def interpolate(F, q, d):
    x, y = q[0] + d[0], q[1] + d[1]
    x1= int(np.floor(x))
    x2 = x1+1
    y1 = int(np.floor(y))
    y2 = y1+1
    
    f1 = (x - x1)*F[:, :, x2, y1] + (x2 - x)*F[:, :, x1, y1]
    f2 = (x - x1)*F[:, :, x2, y2] + (x2 - x)*F[:, :, x1, y2]
    return (y - y1)*f2 + (y2 - y)*f1

def drag_points(w, p, t):
    n = p.shape[0]
    pts = copy.deepcopy(p)
    min_dist = 1e9
    w_best = None
    iter = 0

    w_learned = w[:, :6, :].detach().clone().requires_grad_(True) #only first 6 layers, which correspond to the structure are modified
    w_fixed = w[:, 6:, :].detach().clone().requires_grad_(False)

    F1, _ = gen.synthesis(torch.cat([w_learned, w_fixed], axis=1).to(device))
    F_0 = F1.detach().clone()    

    optimizer = torch.optim.Adam([w_learned.to(device)], lr=2e-3)
    condition = np.all(np.linalg.norm(t-p, ord=2, axis=1) < 3) #checks if the source points are within 3 pixels from target points

    while not condition:
        style = torch.cat([w_learned, w_fixed], axis=1).to(device)
        F, _ = gen.synthesis(style)

        loss = 0
        for i in range(n):
            d = (t[i] - p[i])/(np.linalg.norm(t[i]-p[i], ord=2))
            for q in sigma_ball(p[i], r1, ord=2):              
                f_q = F[:, :, q[0], q[1]].detach()
                f_qmod = interpolate(F, q, d)
                loss += torch.nn.functional.l1_loss(f_qmod, f_q)
        
        #loss += c * torch.norm((F - F_0)*mask, p=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            F, _ = gen.synthesis(style)            

            for i in range(n):
                f_i = F_0[:, :, pts[i][0], pts[i][1]]
                min = 1e9
                min_point = None
                for q in sigma_ball(p[i], r2, ord=1):
                    f_q = interpolate(F, q, [0,0])
                    if torch.norm(f_q - f_i, p=1) < min:
                        min = torch.norm(f_q - f_i, p=1)
                        min_point = q
                p[i] = [int(min_point[0]), int(min_point[1])]

        condition = np.all(np.linalg.norm(t-p, ord=2, axis=1) < 3)

        avg_distance = np.sum(np.linalg.norm(t-p, ord=2, axis=1))/n
        print(f"Iteration {iter}. Average pixel distance is {avg_distance}. The updated points are {p}")

        #if avg_distance > min_dist + 1.5:
        #    return w_best
        
        #elif avg_distance < min_dist:
        #    min_dist = avg_distance
        w_best = torch.cat([w_learned, w_fixed], axis=1).detach()

        iter += 1

    return w_best

def box_centre(i):
    x, y = i[0], i[1]
    centre_x = 61 + x * 100
    centre_y = 61 + y * 100
    return np.array([[centre_x, centre_y]])


if __name__ == "__main__":
    '''w = get_w(z_path)

    w_mod = drag_points(w, pts, tar)

    d = w_mod[:, 0, :] - w[:, 0, :]
    print(torch.norm(d, p=1))
    np.savez('w_mod.npz', w = d.cpu().numpy())

    img1 = G.synthesis(w, noise_mode='const')
    img2 = G.synthesis(w_mod, noise_mode='const')

    img1 = (img1 + 1) * (255/2)
    img1 = img1.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    img2 = (img2 + 1) * (255/2)
    img2 = img2.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    PIL.Image.fromarray(np.hstack([img1, img2]), 'RGB').save('DGAN9.png') '''

boxes = itertools.product(list(range(0, 10)), repeat=2)
w = get_w(z_path)
mv_dict = {}

for i in iter(boxes):
    for j in iter(boxes):
        if i != j:
            p = box_centre(i)
            t = box_centre(j)
            w_mod = drag_points(w, p, t)
            d = w_mod[:, 0, :] - w[:, 0, :]
            mv_dict[(i, j)] = d.cpu().numpy()
        print("1 pair done!")
        np.savez(f'dirn{i}-{j}.npz', mv_dict[(i,j)])

np.savez(f'mv_dict_{number}.npy', mv_dict)