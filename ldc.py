import sys

sys.path.append(".")
from network import DNN
from Utils import *
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0

ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])

### Data Prepareation ###
N_b = 1000  # top wall
N_w = 1000  # no-slip walls
N_c = 10000  # collocation


def getData():
    top_x = np.random.uniform(x_min, x_max, (N_b, 1))
    top_y = np.ones((N_b, 1))
    top_u = np.ones((N_b,1))
    top_v = np.zeros((N_b, 1))
    top_xy = np.concatenate([top_x, top_y], axis=1)
    top_uv = np.concatenate([top_u, top_v], axis=1)

    # wall, u=v=0
    Rwall_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_w, 2))
    dnwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    Lwall_xy = np.random.uniform([x_min, y_min], [x_min, y_max], (N_w, 2))
    Rwall_uv = np.zeros((N_w, 2))
    dnwall_uv = np.zeros((N_w, 2))
    Lwall_uv = np.zeros((N_w, 2))

    # all boundary conds
    xy_bnd = np.concatenate([top_xy, Rwall_xy, dnwall_xy, Lwall_xy], axis=0)
    uv_bnd = np.concatenate([top_uv, Rwall_uv, dnwall_uv, Lwall_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # concatenate all xy for collocation
    xy_col = np.concatenate((xy_col, xy_bnd), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    return xy_col, xy_bnd, uv_bnd


xy_col, xy_bnd, uv_bnd = getData()


class PINN:
    rho = 1
    mu = 0.01

    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=6, n_layer=4, n_node=40, ub=ub, lb=lb).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
        return u, v, p, sig_xx, sig_xy, sig_yy

    def bc_loss(self, xy):
        u, v = self.predict(xy)[0:2]
        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + torch.mean(
            torch.square(v - uv_bnd[:, 1:2])
        )
        return mse_bc

    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p, sig_xx, sig_xy, sig_yy = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), xy, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), xy, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]

        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]

        # continuity equation
        f0 = u_x + v_y

        # navier-stokes equation
        f1 = self.rho * (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = self.rho * (u * v_x + v * v_y) - sig_xy_x - sig_yy_y

        # cauchy stress tensor
        f3 = -p + 2 * self.mu * u_x - sig_xx
        f4 = -p + 2 * self.mu * v_y - sig_yy
        f5 = self.mu * (u_y + v_x) - sig_xy

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5

        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(xy_bnd)
        mse_pde = self.pde_loss(xy_col)
        loss = mse_bc + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss


if __name__ == "__main__":
    pinn = PINN()
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Results/ldc/weight.pt")
    plotLoss(pinn.losses, "./Results/ldc/loss_curve.png", ["BC", "PDE"])
