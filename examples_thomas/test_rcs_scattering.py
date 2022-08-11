import numpy as np
import matplotlib.pyplot as plt

import scipy.constants
import sorts
import numpy.testing as nt

separator = 1/(np.pi*np.sqrt(3)) # = 0.236

diameters = np.logspace(-3, 3, 10000)

is_rayleigh = diameters < separator
is_optical = diameters >= separator

fig = plt.figure()
ax = fig.add_subplot(111)

optical_rcs = np.pi*diameters**2.0/4.0
# ax.loglog(diameters, optical_rcs, "--b")

optical_rcs = optical_rcs[is_optical]
diameters_rcs = diameters[is_optical]


rayleigh_rcs = 9*np.pi**5/4 * diameters**6.0
# ax.loglog(diameters, rayleigh_rcs, "--r")

rayleigh_rcs = rayleigh_rcs[is_rayleigh]
diameters_ray = diameters[is_rayleigh]

ax.loglog(diameters_rcs, optical_rcs, "-b")
ax.loglog(diameters_ray, rayleigh_rcs, "-r")

rcs_sep = np.pi*separator**2.0/4.0

ax.plot([separator, 0], [rcs_sep, rcs_sep], "--k")
ax.plot([separator, separator], [rcs_sep, 0], "--k")

ax.grid()
ax.set_xlim([0.009, 2])
ax.set_ylim([9e-9, 2e1])

ax.set_xlabel("$D/\\lambda$ [$-$]")
ax.set_ylabel("$\\sigma/\\lambda^2$ [$-$]")

plt.show()

# RCS
# wavelength = 1.287 # m
# separator = wavelength/(np.pi*7.11**(1.0/4.0)) # = 0.236

# diameters = np.logspace(-3, 3, 10000)

# is_rayleigh = diameters < separator
# is_optical = diameters >= separator

# fig = plt.figure()
# ax = fig.add_subplot(111)

# optical_rcs = np.pi*diameters**2.0/4.0
# ax.loglog(diameters, optical_rcs, "--b")

# optical_rcs = optical_rcs[is_optical]
# diameters_rcs = diameters[is_optical]


# rayleigh_rcs = np.pi*diameters**2.0*7.11/4.0*(np.pi*diameters/wavelength)**4.0
# ax.loglog(diameters, rayleigh_rcs, "--r")

# rayleigh_rcs = rayleigh_rcs[is_rayleigh]
# diameters_ray = diameters[is_rayleigh]



# ax.loglog(diameters_rcs, optical_rcs, "-b")
# ax.loglog(diameters_ray, rayleigh_rcs, "-r")

# rcs_sep = np.pi*separator**2.0/4.0

# ax.plot([separator, 0], [rcs_sep, rcs_sep], "--k")
# ax.plot([separator, separator], [rcs_sep, 0], "--k")

# ax.grid()
# ax.set_xlim([0.01, 1])
# ax.set_ylim([1e-9, 1e1])

# ax.set_xlabel("$d$ [$m$]")
# ax.set_ylabel("$\\sigma$ [$m^{2}$]")

# plt.show()


# SNR
# wavelength = 1.287 # m
# separator = wavelength/(np.pi*np.sqrt(3)) # = 0.236

# # antenna properties
# gain_tx = 15.0
# gain_rx = 12.5
# power_tx = 22.4e3

# # distance from target
# range_rx_m = 508e3
# range_tx_m = 1200e3

# # other properties
# bandwidth=10
# rx_noise_temp=150.0
# radar_albedo=1.0

# rx_noise = scipy.constants.k*rx_noise_temp*bandwidth
# print(scipy.constants.k)
# diameters = np.logspace(-3, 3, 10000)

# is_rayleigh = diameters < separator
# is_optical = diameters >= separator

# fig = plt.figure()
# ax = fig.add_subplot(111)

# rayleigh_power = (9.0*power_tx*(((gain_tx*gain_rx)*(np.pi**2.0)*(diameters**6.0))/(256.0*(wavelength**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))
# optical_power = (power_tx*(((gain_tx*gain_rx)*(wavelength**2.0)*(diameters**2.0)))/(256.0*(np.pi**2)*(range_rx_m**2.0*range_tx_m**2.0)))

# rayleigh_snr = rayleigh_power*radar_albedo/rx_noise
# optical_snr = optical_power*radar_albedo/rx_noise

# ax.loglog(diameters, optical_snr, "--b")
# ax.loglog(diameters, rayleigh_snr, "--r")

# optical_snr = optical_snr[is_optical]
# rayleigh_snr = rayleigh_snr[is_rayleigh]

# diameters_opt = diameters[is_optical]
# diameters_ray = diameters[is_rayleigh]

# ax.loglog(diameters_opt, optical_snr, "-b")
# ax.loglog(diameters_ray, rayleigh_snr, "-r")

# is_rayleigh = diameters < wavelength/(np.pi*np.sqrt(3.0))
# is_optical = diameters >= wavelength/(np.pi*np.sqrt(3.0))

# rayleigh_power = (9.0*power_tx*(((gain_tx*gain_rx)*(np.pi**2.0)*(diameters**6.0))/(256.0*(wavelength**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))
# optical_power = (power_tx*(((gain_tx*gain_rx)*(wavelength**2.0)*(diameters**2.0)))/(256.0*(np.pi**2)*(range_rx_m**2.0*range_tx_m**2.0)))

# rx_noise = scipy.constants.k*rx_noise_temp*bandwidth
# snr = ((is_rayleigh)*rayleigh_power + (is_optical)*optical_power)*radar_albedo/rx_noise

# ax.loglog(diameters, snr, "-k")


# snr_sep = (9.0*power_tx*(((gain_tx*gain_rx)*(np.pi**2.0)*(separator**6.0))/(256.0*(wavelength**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))

# ax.plot([separator, 0], [snr_sep, snr_sep], "--k")
# ax.plot([separator, separator], [snr_sep, 0], "--k")

# ax.grid()
# ax.set_xlim([0.01, 1])
# ax.set_ylim([1e-9, 1e1])

# ax.set_xlabel("$d$ [$m$]")
# ax.set_ylabel("$\\sigma$ [$m^{2}$]")

# plt.show()

