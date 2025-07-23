import numpy as np
from infrapy.thermoelasticity import lock_in_analysis, thermoelastic_calibration, get_strain

## Using lock_in_analysis (the Lock-In Amplifier method)

# Example thermal data: [frames, height, width]
thermal_data = np.random.rand(1000, 64, 64)  # simulated data

fs = 1000  # sampling frequency (Hz)
fl = 50    # loading frequency (Hz)

# Call lock_in_analysis
magnitude, phase, residuals = lock_in_analysis(thermal_data, fs, fl, return_residuals=True)

print("Magnitude shape:", magnitude.shape)
print("Phase shape:", phase.shape)

## Using thermoelastic_calibration - Get thermoelastic coefficient from known material
km = thermoelastic_calibration(method='material', material='steel')
print(f"Thermoelastic coefficient for steel: {km:.3e} Pa^-1")


## Using thermoelastic_calibration - Compute strain from strain gauge data (direct strain)
# Example strain gauge data: time series x gauges
strain_gauge_data = np.random.rand(1000, 3)  # 3-gauge rosette data

strain_equivalent = thermoelastic_calibration(
    method='direct_strain',
    strain_data=strain_gauge_data,
    strain_config='120'  # or '90' depending on rosette geometry
)

print("Equivalent strain from strain gauges:", strain_equivalent)


## Using thermoelastic_calibration - Compute thermoelastic coefficient from thermal and strain gauge data
thermal_data = np.random.rand(1000, 64, 64)  # Thermal images
strain_gauge_data = np.random.rand(1000, 3)  # Strain gauge data

fs = 1000  # Sampling frequency (Hz)
fl = 50    # Load frequency (Hz)
E = 210e9  # Young's modulus for steel (Pa)
nu = 0.3   # Poisson's ratio for steel
location = (10, 10, 5, 5)  # ROI in thermal images where strain gauge is located

km = thermoelastic_calibration(
    method='strain_gauge',
    thermal_data=thermal_data,
    fs=fs,
    fl=fl,
    E=E,
    nu=nu,
    strain_data=strain_gauge_data,
    strain_config='120',
    location=location
)

print(f"Calibrated thermoelastic coefficient: {km:.3e} Pa^-1")

## Integrate lock_in_analysis inside thermoelastic calibration (custom approach)
# Replace 'pyLIA.LIA' with your lock_in_analysis inside from_strain_gauge

magnitude, phase, residuals = lock_in_analysis(thermal_data, fs, fl, return_residuals=True)
mag_roi = np.mean(magnitude[location[1]:location[1]+location[3], location[0]:location[0]+location[2]])

strain = get_strain(strain_gauge_data, configuration='120')

km_custom = (mag_roi * (1 - nu)) / (E * strain)

print(f"Calibrated thermoelastic coefficient (custom LIA): {km_custom:.3e} Pa^-1")
