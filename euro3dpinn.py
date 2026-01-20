# now let's do another plot but as in the previous one, with magnetic field vectors along the trajectories of all 3 flybys.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import plasma

# Load ESPRH spherical data for all 3 flybys
df1 = pd.read_csv("data/ORB04_EUR_ESPRH.csv", parse_dates=["SPACECRAFT EVENT TIME"])
df2 = pd.read_csv("data/ORB11_EUR_ESPRH.csv", parse_dates=["SPACECRAFT EVENT TIME"])
df3 = pd.read_csv("data/ORB14_EUR_ESPRH.csv", parse_dates=["SPACECRAFT EVENT TIME"])

# Down-sample for legibility
stride1 = max(len(df1) // 2000, 1)
d1 = df1.iloc[::stride1].copy()
stride2 = max(len(df2) // 2000, 1)
d2 = df2.iloc[::stride2].copy()
stride3 = max(len(df3) // 2000, 1)
d3 = df3.iloc[::stride3].copy()

# Spherical -> Cartesian positions for flyby 1
theta = np.deg2rad(90.0 - d1["LATITUDE"].to_numpy())
phi = np.deg2rad(d1["LONGITUDE"].to_numpy())
r = d1["RANGE"].to_numpy()
x1 = r * np.sin(theta) * np.cos(phi)
y1 = r * np.sin(theta) * np.sin(phi)
z1 = r * np.cos(theta)

# Spherical -> Cartesian positions for flyby 2
theta2 = np.deg2rad(90.0 - d2["LATITUDE"].to_numpy())
phi2 = np.deg2rad(d2["LONGITUDE"].to_numpy())
r2 = d2["RANGE"].to_numpy()
x2 = r2 * np.sin(theta2) * np.cos(phi2)
y2 = r2 * np.sin(theta2) * np.sin(phi2)
z2 = r2 * np.cos(theta2)

# Spherical -> Cartesian positions for flyby 3
theta3 = np.deg2rad(90.0 - d3["LATITUDE"].to_numpy())
phi3 = np.deg2rad(d3["LONGITUDE"].to_numpy())
r3 = d3["RANGE"].to_numpy()
x3 = r3 * np.sin(theta3) * np.cos(phi3)
y3 = r3 * np.sin(theta3) * np.sin(phi3)
z3 = r3 * np.cos(theta3)

# Spherical components -> Cartesian vectors for flyby 1
Br1 = d1["BR"].to_numpy()
Btheta1 = d1["BTHETA"].to_numpy()
Bphi1 = d1["BPHI"].to_numpy()
Bx1 = Br1 * np.sin(theta) * np.cos(phi) + Btheta1 * np.cos(theta) * np.cos(phi) - Bphi1 * np.sin(phi)
By1 = Br1 * np.sin(theta) * np.sin(phi) + Btheta1 * np.cos(theta) * np.sin(phi) + Bphi1 * np.cos(phi)
Bz1 = Br1 * np.cos(theta) - Btheta1 * np.sin(theta)
Bmag1 = np.sqrt(Bx1**2 + By1**2 + Bz1**2)
Bx1_u = Bx1 / Bmag1
By1_u = By1 / Bmag1
Bz1_u = Bz1 / Bmag1

# Spherical components -> Cartesian vectors for flyby 2
Br2 = d2["BR"].to_numpy()
Btheta2 = d2["BTHETA"].to_numpy()
Bphi2 = d2["BPHI"].to_numpy()
Bx2 = Br2 * np.sin(theta2) * np.cos(phi2) + Btheta2 * np.cos(theta2) * np.cos(phi2) - Bphi2 * np.sin(phi2)
By2 = Br2 * np.sin(theta2) * np.sin(phi2) + Btheta2 * np.cos(theta2) * np.sin(phi2) + Bphi2 * np.cos(phi2)
Bz2 = Br2 * np.cos(theta2) - Btheta2 * np.sin(theta2)
Bmag2 = np.sqrt(Bx2**2 + By2**2 + Bz2**2)
Bx2_u = Bx2 / Bmag2
By2_u = By2 / Bmag2
Bz2_u = Bz2 / Bmag2

# Spherical components -> Cartesian vectors for flyby 3
Br3 = d3["BR"].to_numpy()
Btheta3 = d3["BTHETA"].to_numpy()
Bphi3 = d3["BPHI"].to_numpy()
Bx3 = Br3 * np.sin(theta3) * np.cos(phi3) + Btheta3 * np.cos(theta3) * np.cos(phi3) - Bphi3 * np.sin(phi3)
By3 = Br3 * np.sin(theta3) * np.sin(phi3) + Btheta3 * np.cos(theta3) * np.sin(phi3) + Bphi3 * np.cos(phi3)
Bz3 = Br3 * np.cos(theta3) - Btheta3 * np.sin(theta3)
Bmag3 = np.sqrt(Bx3**2 + By3**2 + Bz3**2)
Bx3_u = Bx3 / Bmag3
By3_u = By3 / Bmag3
Bz3_u = Bz3 / Bmag3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection="3d")

# Europa sphere (radius = 1 R_Europa)
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.25, linewidth=0, zorder=0)

norm1 = Normalize(vmin=Bmag1.min(), vmax=Bmag1.max())
colors1 = plasma(norm1(Bmag1))
ax.quiver(
    x1, y1, z1,
    Bx1_u, By1_u, Bz1_u,
    length=0.12, normalize=False, colors=colors1, linewidth=0.4
)
norm2 = Normalize(vmin=Bmag2.min(), vmax=Bmag2.max())
colors2 = plasma(norm2(Bmag2))
ax.quiver(
    x2, y2, z2,
    Bx2_u, By2_u, Bz2_u,
    length=0.12, normalize=False, colors=colors2, linewidth=0.4
)
norm3 = Normalize(vmin=Bmag3.min(), vmax=Bmag3.max())
colors3 = plasma(norm3(Bmag3))
ax.quiver(
    x3, y3, z3,
    Bx3_u, By3_u, Bz3_u,
    length=0.12, normalize=False, colors=colors3, linewidth=0.4
)
mappable = plt.cm.ScalarMappable(norm=norm1, cmap=plasma)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label("|B| [nT]")
ax.set_xlabel("X [R_Europa]")
ax.set_ylabel("Y [R_Europa]")
ax.set_zlabel("Z [R_Europa]")
ax.set_title("Galileo Europa Flyby Magnetic Field Vectors (All 3 Flybys with Europa)")

# Equal-ish aspect
lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
center = lims.mean(axis=1)
radius = (lims[:,1] - lims[:,0]).max() / 2
ax.set_xlim3d([center[0]-radius, center[0]+radius])
ax.set_ylim3d([center[1]-radius, center[1]+radius])
ax.set_zlim3d([center[2]-radius, center[2]+radius])
plt.show()