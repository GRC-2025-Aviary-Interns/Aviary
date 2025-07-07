import numpy as np
import matplotlib.pyplot as plt


def PropDataReader():
    """A function to read in propulsion data to ndarrays and apply reasonable
        restrictions to training data
        Note that prop_xt.dat file units are given in [m deg rev/s m/s]

    Returns:
        ndarrays: x data, thrust and power coefficients
    """

    # Initializing ndarrays
    xt = np.ndarray((0, 4))
    ct = np.ndarray((0, 1))
    cp = np.ndarray((0, 1))

    # Opening data files
    # CHANGE THESE TO THE FILE PATH
    fx = open("aviary\subsystems\propulsion\\rc_electric\Parsing\prop_xt.dat", "r")
    fy = open("aviary\subsystems\propulsion\\rc_electric\Parsing\prop_yt.dat", "r")

    # Looping through lines of .dat file
    for xline, yline in zip(fx.readlines(), fy.readlines()):

        # Converting line into array
        xlineArr = np.array([float(i) for i in xline.strip().split(" ")])
        ylineArr = np.array([float(i) for i in yline.strip().split(" ")])

        # Conversion factors
        in2m = 0.0254
        rpm2rps = 1 / 60

        # Restricting data to reasonable bounds
        if (
            xlineArr[3] < 50  # V < 50 m/s
            and xlineArr[0] / in2m < 23  # D < 23 in
            and xlineArr[0] / in2m > 12  # D > 14
            and xlineArr[2] / rpm2rps
            < 150000 / (xlineArr[0] / in2m)  # RPM limits suggested by APC
        ):
            # If within limits, add to returning arrays
            xt = np.vstack([xt, xlineArr])
            ct = np.vstack([ct, ylineArr[0]])
            cp = np.vstack([cp, ylineArr[1]])

    return xt, ct, cp
