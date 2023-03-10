import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# import time
# import sys
# sys.path.append("C:\\Users\\JONAF\\projects\\fea-learning\\fea-engine")
from simulation.sim_funcs import *
from time import time




def bulge_simulation(C, p_max, h=2.0, inc=100, tolerance=0.15, Radius=50.0):
    n = 10 # elements
    b = 20 # mm
    Total_Nodes = 2 * n + 1
    IS = 1.0
    DOF_BC = np.zeros(shape=(Total_Nodes * 2, 1))
    DOF_BC[0, 0] = 1
    DOF_BC[-2, 0] = 1
    DOF_BC[-1, 0] = 1

    r = np.zeros(shape=(Total_Nodes, 1))
    z = np.zeros_like(r)
    R = b/2 + ((Radius * 2)**2)/(8 * b)
    alpha = np.arcsin(Radius/R)*180/np.pi
    L = alpha * np.pi * R / 180
    hh = L / (2 * n)
    le = hh * 2
    r_arc = np.zeros(shape=(21,))
    z_arc = np.zeros_like(r_arc)
    for j in range(len(r_arc)-1):
        z_arc[j + 1] = (1 / 2) * np.sqrt((-(
                    4 * (R * z_arc[j] - (1 / 2) * hh ** 2 - (1 / 2) * r_arc[j] ** 2 - (1 / 2) * z_arc[j] ** 2)) * (
                                                      R - z_arc[j]) * 2 * np.sqrt(
            r_arc[j] ** 2 * ((1 / 2) * r_arc[j] ** 2 + (hh - z_arc[j]) * (-(1 / 2) * hh + R - (1 / 2) * z_arc[j])) * (
                        -(1 / 2) * r_arc[j] ** 2 + (hh + z_arc[j]) * ((1 / 2) * hh + R - (1 / 2) * z_arc[j]))) + z_arc[
                                              j] ** 6 - 6 * R * z_arc[j] ** 5 + (
                                                      13 * R ** 2 + 2 * hh ** 2 + r_arc[j] ** 2) * z_arc[j] ** 4 + (
                                                      -12 * R ** 3 + (-8 * hh ** 2 - 4 * r_arc[j] ** 2) * R) * z_arc[
                                              j] ** 3 + (4 * R ** 4 + (
                    10 * hh ** 2 + 6 * r_arc[j] ** 2) * R ** 2 + hh ** 4 + 4 * hh ** 2 * r_arc[j] ** 2 - r_arc[
                                                             j] ** 4) * z_arc[j] ** 2 - 4 * R * (
                                                      (hh ** 2 + r_arc[j] ** 2) * R ** 2 + (
                                                          1 / 2) * hh ** 4 + 2 * hh ** 2 * r_arc[j] ** 2 - (1 / 2) *
                                                      r_arc[j] ** 4) * z_arc[j] + (
                                                      hh ** 4 + 6 * hh ** 2 * r_arc[j] ** 2 + r_arc[j] ** 4) * R ** 2 -
                                          r_arc[j] ** 2 * (hh - r_arc[j]) ** 2 * (hh + r_arc[j]) ** 2) / (
                                                     R ** 2 - 2 * R * z_arc[j] + r_arc[j] ** 2 + z_arc[j] ** 2) ** 2) + \
                       z_arc[j]

        r_arc[j + 1] = np.sqrt(2 * R * z_arc[j + 1] - z_arc[j + 1] ** 2)
    zz_arc = z_arc[-1] - z_arc
    Nid = np.vstack([r_arc, zz_arc]).T

    Elecon = np.zeros(shape=(n, 3), dtype=int)
    k = 0
    for i in range(n):
        Elecon[i, 0] = k
        Elecon[i, 1] = k + 1
        Elecon[i, 2] = k + 2
        k += 2

    ue = np.zeros(shape=(6, n))
    for i in range(n):
        ue[0, i] = Nid[Elecon[i, 0], 0]
        ue[1, i] = Nid[Elecon[i, 0], 1]
        ue[2, i] = Nid[Elecon[i, 1], 0]
        ue[3, i] = Nid[Elecon[i, 1], 1]
        ue[4, i] = Nid[Elecon[i, 2], 0]
        ue[5, i] = Nid[Elecon[i, 2], 1]
    Re = ue.copy()

    for i in range(n):
        ue[0, i] = Nid[Elecon[i, 0], 0] * IS
        ue[1, i] = Nid[Elecon[i, 0], 1]
        ue[2, i] = Nid[Elecon[i, 1], 0] * IS
        ue[3, i] = Nid[Elecon[i, 1], 1]
        ue[4, i] = Nid[Elecon[i, 2], 0] * IS
        ue[5, i] = Nid[Elecon[i, 2], 1]
    p = 0
    u_inc = np.zeros((Total_Nodes*2, inc))
    u_inc_r = np.zeros((2 * n + 1, inc))
    u_inc_z = np.zeros_like(u_inc_r)
    p_vec = np.linspace(p_max/inc, p_max, inc)
    for zz, p in enumerate(p_vec):
        Flag = 0
        while Flag == 0:
            G_elem = bulge_G_elem(C[0], C[1], C[2], le, h, n, p, ue, Re)
            KT_elem = bulge_Stiff_elem_No_p(C[0], C[1], C[2], le, h, n, p, ue, Re)
            G = bulge_Assem_G(G_elem, Elecon, n, Total_Nodes, DOF_BC)
            KT = bulge_Assem_KT(KT_elem, Elecon, n, Total_Nodes, DOF_BC)
            # ans = np.linalg.lstsq(KT, G.squeeze(), rcond=None)
            # du = -ans[0]
            du = -np.linalg.solve(KT, G.squeeze())
            if abs(du.T @ du) <= tolerance:
                Flag = 1

            due = bulge_du_to_due(n, Elecon, du)
            ue = ue + due
        u = bulge_ue_to_u(n, Total_Nodes, Elecon, ue)
        u_inc[:, zz] = u[:, 0]
        u_inc_r[:, zz] = u_inc[0::2, zz]
        u_inc_z_check = u_inc[1::2, zz]
        if u_inc_z_check[-1] < -10:
            break
        u_inc_z[:, zz] = u_inc_z_check

    bulge_height = u_inc_z[0, :] - b
    valid_inds = bulge_height >= 0
    
    
    return bulge_height[valid_inds], p_vec[valid_inds]


def main():
    C = np.array([0.000533400000000000, 2.19600000000000e-05, 8.55300000000000e-08])
    bulge_height, p_vec = bulge_simulation(C, p_max=2.6E-4, h=7.3, inc=100, tolerance=0.1, Radius=50.0)
    fig = px.line(x=bulge_height, y=p_vec)
    fig.show()
    return


if __name__ == '__main__':
    main()