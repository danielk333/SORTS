import sorts
import numpy as np

from sorts import frames

lon = 0
lat = 0
alt = 0

ecef = frames.azel_to_ecef(lat, lon, alt, 90, 45, radians=False)

print("ecef = ", ecef)