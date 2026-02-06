import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sorts

grid_size = [10, 10, 10]
R_earth = 6371e3


pop = sorts.population.orbit_grid(
    semi_major_axis_samples=np.linspace(R_earth + 300e3, R_earth + 1200e3, num=grid_size[0]),
    eccentricity_samples=np.array([0]),
    inclination_samples=np.linspace(60, 120, num=grid_size[1]),
    argument_of_periapsis_samples=np.array([0]),
    longitude_of_ascending_node_samples=np.array([0]),
    mean_anomaly_samples=np.array([0]),
    diameter_samples=10 ** np.linspace(-2, 1, num=grid_size[2]),
)

print(pop.print(np.arange(10)))

with tempfile.TemporaryDirectory() as td:
    tempd = Path(td)
    print("created temporary directory, saving pop", tempd)
    pop.save(tempd / "pop.h5")
    pop2 = sorts.Population.load(tempd / "pop.h5")

    print("loaded population:")
    print(pop2.print(np.arange(10)))

prop = sorts.propagator.Sgp4(
    settings=sorts.propagator.Sgp4Settings(
        out_frame="ITRS",
    )
)
spobj = pop.get_object(0)
print(spobj)

t = np.arange(0, 24 * 3600, 120)
states = prop.propagate(spobj, t)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(221)
ax.plot(pop.data["a"], pop.data["i"])

ax = fig.add_subplot(223)
ax.plot(pop.data["a"], pop.data["d"])

ax = fig.add_subplot(122, projection="3d")
ax.plot(states[0, :], states[1, :], states[2, :])

plt.show()
