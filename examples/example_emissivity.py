"""
This is a basic use case of InfraPy.
"""

import infrapy

## Example 1

# Measured by a calibrated IR camera pointed at a heated aluminum plate (ε = 0.92).

# Apparent radiance from camera (after calibration), e.g., using NETD and LUTs
measured_radiance = 2.15  # W/sr/m² over 7.5–13 µm band
emissivity = 0.92
ambient_temp = 293.15  # Room temperature in K

# Apply emissivity correction
true_radiance = infrapy.correct_emissivity(measured_radiance, emissivity=emissivity, ambient_temp_K=ambient_temp)

# Convert to temperature
estimated_temp = infrapy.temperature_from_radiance(true_radiance, method="band")
print(f"Estimated true surface temperature: {estimated_temp:.2f} K ≈ {estimated_temp - 273.15:.2f} °C")

## Example 2

# Compare predicted radiance with actual camera reading when viewing a blackbody source at 50 °C.

blackbody_temp = 323.15  # 50 °C in K
emissivity = 0.98  # high-precision blackbody

# Predict radiance
ideal_radiance = infrapy.radiance_from_temperature(blackbody_temp, method="band")[0]

# Simulate camera seeing slightly less due to imperfect optics
measured_radiance = ideal_radiance * 0.97  # e.g., losses due to window

# Estimate temperature from slightly reduced radiance
recovered_temp = infrapy.temperature_from_radiance(measured_radiance, method="band")
print(f"Blackbody: True T = 50°C → Measured T ≈ {recovered_temp - 273.15:.2f} °C")


## Example 3

# Steel bar at ~600 °C measured in a lab with emissivity 0.70 and ambient temperature 22 °C.

emissivity = 0.70
ambient_temp_K = 295.15  # 22 °C
true_temp_K = 873.15  # 600 °C

# What would the camera see?
true_rad = infrapy.radiance_from_temperature(true_temp_K, method="band")[0]
apparent_rad = emissivity * true_rad + (1 - emissivity) * infrapy.radiance_from_temperature(ambient_temp_K, method="band")[0]

print(f"Simulated apparent radiance: {apparent_rad:.2f} W/sr/m²")

# Invert to get back the temperature (without emissivity correction)
apparent_temp = infrapy.temperature_from_radiance(apparent_rad, method="band")
print(f"Apparent temperature without correction: {apparent_temp - 273.15:.2f} °C")

# Now apply correction and recover the true surface temp
corrected_rad = infrapy.correct_emissivity(apparent_rad, emissivity=emissivity, ambient_temp_K=ambient_temp_K)
corrected_temp = infrapy.temperature_from_radiance(corrected_rad, method="band")
print(f"Recovered temperature after correction: {corrected_temp - 273.15:.2f} °C")
