
from ldc import PINN, x_min, x_max, y_min, y_max
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Results/ldc/weight.pt"))

x = np.linspace(x_min,x_max,100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)

xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy, dtype=torch.float32).to(device)

x_top = np.linspace(x_min,x_max,100).reshape(-1,1)
y_top = np.ones_like(x_top).reshape(-1,1)
xy_top = np.concatenate([x_top, y_top], axis=1)
xy_top = torch.tensor(xy_top, dtype=torch.float32).to(device)

with torch.no_grad():
    u, v, p, sig_xx, sig_xy, sig_yy = pinn.predict(xy)
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    p = p.cpu().numpy()
    u_top, v_top, _, _, _, _ = pinn.predict(xy_top)
    u_top = u_top.cpu().numpy()
    v_top = v_top.cpu().numpy()

plt.figure()
plt.contourf(x.reshape(100,100), y.reshape(100,100), p.reshape(100,100), cmap=cm.jet)
plt.colorbar()
plt.title(f"Pressure Field")
plt.savefig('Results/ldc/Pressure_Field.png')
plt.figure()
plt.contourf(x.reshape(100,100), y.reshape(100,100), u.reshape(100,100), cmap=cm.jet)
plt.colorbar()
plt.title(f"u Field")
plt.savefig('Results/ldc/u_Field.png')
plt.figure()
plt.contourf(x.reshape(100,100), y.reshape(100,100), v.reshape(100,100), cmap=cm.jet)
plt.colorbar()
plt.title(f"v Field")
plt.savefig('Results/ldc/v_Field.png')
plt.figure()
plt.subplot(1,2,1)
plt.plot(x_top, u_top)
plt.title("u-profile at top wall")
plt.subplot(1,2,2)
plt.plot(x_top, v_top)
plt.title("v-profile at top wall")
plt.savefig('Results/ldc/Top_Wall_Results.png')
