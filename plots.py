import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
def extract_estimates(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) < 4:
            raise ValueError("File does not contain enough lines.")
        
        average_code_phase_estimate_gps = list(map(float, lines[0].strip().split()))
        average_code_phase_estimate_navic   = list(map(float, lines[1].strip().split()))
        average_doppler_estimate_gps    = list(map(float, lines[2].strip().split()))
        average_doppler_estimate_navic      = list(map(float, lines[3].strip().split()))
        
        return (average_code_phase_estimate_gps,
                average_code_phase_estimate_navic,
                average_doppler_estimate_gps,
                average_doppler_estimate_navic)

actual_dopplers = [1502.36, 1567, 4501.2, 6768.69, 8901.45, 1567, 1596, 1100.2, 6709.1, 7891.2]
# actual_dopplers = [4501.2]*10
actual_code_phase = [19, 1, 15, 2, 33, 74, 5, 56, 37, 28]
# Print to check

nfiles = 0

filename = "estimates_good_doppler_-20.txt"
gps_code_phases_m20, navic_code_phases_m20, gps_dopplers_m20, navic_dopplers_m20 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_-10.txt"
gps_code_phases_m10, navic_code_phases_m10, gps_dopplers_m10, navic_dopplers_m10 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_-5.txt"
gps_code_phases_m5, navic_code_phases_m5, gps_dopplers_m5, navic_dopplers_m5 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_-3.txt"
gps_code_phases_m3, navic_code_phases_m3, gps_dopplers_m3, navic_dopplers_m3 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_1.txt"
gps_code_phases_1, navic_code_phases_1, gps_dopplers_1, navic_dopplers_1 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_3.txt"
gps_code_phases_3, navic_code_phases_3, gps_dopplers_3, navic_dopplers_3 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_5.txt"
gps_code_phases_5, navic_code_phases_5, gps_dopplers_5, navic_dopplers_5 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_10.txt"
gps_code_phases_10, navic_code_phases_10, gps_dopplers_10, navic_dopplers_10 = extract_estimates(filename)
nfiles+=1

filename = "estimates_good_doppler_20.txt"
gps_code_phases_20, navic_code_phases_20, gps_dopplers_20, navic_dopplers_20 = extract_estimates(filename)
nfiles+=1

# satellite number = i+1
for i in range(0,10):
    delta_doppler_navic = []
    delta_doppler_gps = []
    # delta_doppler_navic contains the difference between the actual doppler and the navic doppler as the function of snr value for a particular satellite
    # delta_doppler_gps contains the difference between the actual doppler and the gps doppler as the function of snr value for a particular satellite
    # delta_doppler_navic
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_m20[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_m20[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_m10[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_m10[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_m5[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_m5[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_m3[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_m3[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_1[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_1[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_3[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_3[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_5[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_5[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_10[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_10[i]))
    delta_doppler_gps.append(abs(-actual_dopplers[i]-gps_dopplers_20[i]))
    delta_doppler_navic.append(abs(-actual_dopplers[i]-navic_dopplers_20[i]))
    snr = [-20,-10, -5, -3, 1, 3, 5, 10,20]
    # Plotting the results
    # plt.figure(figsize=(10, 5))
    # # delta_doppler_gps = gaussian_filter1d(delta_doppler_gps, sigma=1)
    # # delta_doppler_navic = gaussian_filter1d(delta_doppler_navic, sigma=1)
    # plt.plot(snr, delta_doppler_navic, marker='o', label='NavIC Doppler Deviation')
    # plt.plot(snr, delta_doppler_gps, marker='o', label='GPS Doppler Deviation')
    # plt.title(f'Doppler Deviation for Satellite {i+1}')
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Deviation (Hz)')
    # plt.grid()
    # plt.legend()
    # plt.xticks(snr)
    # # plt.show()
    # # Save the plot
    # caption = f'Doppler_Deviation_Satellite_{i+1}.jpeg'
    # plt.savefig(caption, bbox_inches='tight', dpi=300)
    # plt.close()  # Close the figure to free memory
#     # break

# Initialize a matrix to store which estimate is better (1 for NavIC, 0 for GPS)
heatmap_data = np.zeros((10, 9))  # 10 satellites, 9 SNR values

snr = [-20, -10, -5, -3, 1, 3, 5, 10, 20]

# Create dictionaries to map SNR values to their corresponding variables
navic_dopplers = {
    -20: navic_dopplers_m20, -10: navic_dopplers_m10, -5: navic_dopplers_m5,
    -3: navic_dopplers_m3, 1: navic_dopplers_1, 3: navic_dopplers_3,
    5: navic_dopplers_5, 10: navic_dopplers_10, 20: navic_dopplers_20
}
gps_dopplers = {
    -20: gps_dopplers_m20, -10: gps_dopplers_m10, -5: gps_dopplers_m5,
    -3: gps_dopplers_m3, 1: gps_dopplers_1, 3: gps_dopplers_3,
    5: gps_dopplers_5, 10: gps_dopplers_10, 20: gps_dopplers_20
}

for i in range(0, 10):
    for j, snr_value in enumerate(snr):
        # Calculate absolute errors for NavIC and GPS
        navic_error = abs(-actual_dopplers[i] - navic_dopplers[snr_value][i])
        gps_error = abs(-actual_dopplers[i] - gps_dopplers[snr_value][i])
        
        # Determine which estimate is better
        heatmap_data[i, j] = 1 if navic_error < gps_error else 0

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", xticklabels=snr, yticklabels=[f"Sat {i+1}" for i in range(10)],
            cbar_kws={'label': 'Better Estimate (1=NavIC, 0=GPS)'})
plt.title("Heatmap of Better Doppler Estimate (NavIC vs GPS)")
plt.xlabel("SNR (dB)")
plt.ylabel("Satellite Number")
plt.show()
