import math
import cmath
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def PRN(sv):
        SV = {
            1: [2,6], 2: [3,7], 3: [4,8], 4: [5,9], 5: [1,9], 6: [2,10], 7: [1,8], 8: [2,9],
            9: [3,10], 10: [2,3], 11: [3,4], 12: [5,6], 13: [6,7], 14: [7,8], 15: [8,9],
            16: [9,10], 17: [1,4], 18: [2,5], 19: [3,6], 20: [4,7], 21: [5,8], 22: [6,9],
            23: [1,3], 24: [4,6], 25: [5,7], 26: [6,8], 27: [7,9], 28: [8,10], 29: [1,6],
            30: [2,7], 31: [3,8], 32: [4,9],
        }
        def shift(register, feedback, output):
            out = [register[i-1] for i in output]
            if len(out) > 1:
                out = sum(out) % 2
            else:
                out = out[0]
            fb = sum([register[i-1] for i in feedback]) % 2
            for i in reversed(range(len(register[1:]))):
                register[i+1] = register[i]
            register[0] = fb
            return out

        G1 = [1 for i in range(10)]
        G2 = [1 for i in range(10)]
        ca = []
        for i in range(1023):
            g1 = shift(G1, [3,10], [10])
            g2 = shift(G2, [2,3,6,8,9,10], SV[sv])
            ca.append((g1 + g2) % 2)
        return ca

def add_gaussian_noise(signal, snr_db):
    N = len(signal)

    signal_power = sum(abs(s)**2 for s in signal) / N

    snr_linear = 10 ** (snr_db / 10.0)

    noise_power = signal_power / snr_linear
    noise_stddev = np.sqrt(noise_power / 2.0)  # For real and imaginary parts

    noise_real = np.random.normal(0.0, noise_stddev, N)
    noise_imag = np.random.normal(0.0, noise_stddev, N)
    noise = [complex(r, im) for r, im in zip(noise_real, noise_imag)]

    noisy_signal = [s + complex(r, im) for s, r, im in zip(signal, noise_real, noise_imag)]

    return noisy_signal

def upsample(prn, time_prn, upsample_freq):
    prn_upsample = []
    total_samples = int(time_prn * upsample_freq)

    for i in range(total_samples):
        index = math.ceil(((i + 1) * len(prn)) / total_samples) - 1
        prn_upsample.append(prn[index])

    return prn_upsample

def IntegerDelay_Navic(signal, k, sv):
    with open(f"Navic_PRN_{sv}.txt", "r") as file:
        prn = list(map(int, file.read().strip().split()))
        prn = upsample(prn, 5 * 1e-3, 4 * 1e6)
    temp = [float(prn[i]) for i in range(len(prn) - k, len(prn))]

    for i in range(len(signal) - 1, k - 1, -1):
        signal[i] = signal[i - k]

    for i in range(k):
        signal[i] = temp[i]

def IntegerDelay_gps(signal, k, sv):
    prn = PRN(sv)
    prn = upsample(prn, 1e-3, 4e6)
    temp = [float(prn[i]) for i in range(len(prn) - k, len(prn))]

    for i in range(len(signal) - 1, k - 1, -1):
        signal[i] = signal[i - k]

    for i in range(k):
        signal[i] = temp[i]

def apply_doppler(signal, fd, sampling_rate):
    output = []
    for i in range(len(signal)):
        doppler = cmath.rect(1.0, 2 * math.pi * fd * i / sampling_rate)  # polar form
        signal_value = complex(signal[i])
        output.append(signal_value * doppler)
    return output

def apply_doppler_complex(signal: list[complex], fd: float, sampling_rate: float) -> list[complex]:
    n = len(signal)
    t = np.arange(n) / sampling_rate
    doppler = np.exp(1j * 2 * np.pi * fd * t)
    return list(np.array(signal) * doppler)

def correlate(x: list[complex], y: list[complex]) -> list[complex]:
    n = len(x) + len(y) - 1
    corr = fftconvolve(x, y[::-1], mode='valid')
    return corr[:len(x)]  # Return only the valid part if desired

def calculate_threshold(signal: list[complex], sample_rate: float, doppler: float, spread_preamble: list[complex]) -> float:
    max_corr = -1e9
    best_index = 0
    best_doppler = 0.0
    code_phase_est = 0
    init_doppler_estimate = 0.0
    preamble_len = len(spread_preamble)
    data_len = len(signal)
    signal = apply_doppler_complex(signal, -doppler, sample_rate)
    corr = correlate(signal, spread_preamble)
    plt.plot(np.arange(len(corr)), np.real(corr), label="Real Part")
    plt.plot(np.arange(len(corr)), np.imag(corr), label="Imaginary Part")
    # plt.plot(np.arange(len(corr)), np.abs(corr), label="Magnitude")
    plt.legend()
    plt.title("Correlation Plot")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Value")
    plt.grid()
    plt.show()
    threshold = np.mean(np.abs(np.imag(corr)))
    return threshold*10


def code_phase_doppler_estimate(data: list[complex], 
                                sample_rate: float, 
                                doppler_values: list[float], 
                                spread_preamble: list[complex],threshold) -> list[float]:
    min_avg_corr = 1e7
    best_index = 0
    best_doppler = 0.0
    doppler = doppler_values[0]
    code_phase_est = 0
    init_doppler_estimate = 0.0

    preamble_len = len(spread_preamble)
    data_len = len(data)

    # Loop over Doppler shifts from -5 to 5 with step 0.1
    rerun = True
    while rerun:
        for doppler_shift in np.arange(-5, 6,0.1):  # -5 to 5 inclusive
            shifted_data = apply_doppler_complex(data, -doppler - doppler_shift, sample_rate)
            corr = correlate(shifted_data, spread_preamble)
            max_corr_temp = max(abs(corr[:preamble_len]))
            max_index = np.argmax(abs(corr[:preamble_len]))
            avg_corr = -1e9
            if np.all(abs(np.imag(corr))<threshold):
                rerun = False
                avg_corr = np.mean(np.abs(np.imag(corr)))
                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    best_index = max_index
                    best_doppler = doppler_shift
                    best_corr = corr
            if rerun == True:
                threshold = threshold*1.1

    plt.plot(np.arange(len(best_corr)), np.real(best_corr))
    plt.plot(np.arange(len(best_corr)), np.imag(best_corr))
    # plt.plot(np.arange(len(best_corr)), abs(best_corr), label="Magnitude")
    plt.legend(["Real Part", "Imaginary Part"])
    plt.title(f"Correlation with Doppler Shift: {best_doppler:.2f} Hz")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation Magnitude")
    plt.grid()

    plt.show()
    code_phase_est = best_index
    init_doppler_estimate = best_doppler

    print("Code Phase Estimate:", code_phase_est)
    print("Doppler Estimate:", -doppler - init_doppler_estimate)
    print("Max Correlation:", avg_corr)

    return [float(code_phase_est), np.round(-doppler - init_doppler_estimate,3)]

def write_estimates_to_file(
    snr: float,
    avg_code_phase_gps: list[float],
    avg_code_phase_navic: list[float],
    avg_doppler_gps: list[float],
    avg_doppler_navic: list[float]
):
    filename = f"estimates_good_doppler_{snr}.txt"
    try:
        with open(filename, 'w') as outfile:
            def write_vector(vec):
                outfile.write(' '.join(map(str, vec)) + '\n')
            
            write_vector(avg_code_phase_gps)
            write_vector(avg_code_phase_navic)
            write_vector(avg_doppler_gps)
            write_vector(avg_doppler_navic)

    except IOError:
        print(f"Error opening file: {filename}")



snr_values = [-3, 1, 3, 5, 10, 20]
for snr in snr_values:
    average_code_phase_estimate_navic = [0.0] * 10
    average_code_phase_estimate_gps = [0.0] * 10
    average_doppler_estimate_navic = [0.0] * 10
    average_doppler_estimate_gps = [0.0] * 10
    num_iterations = 10

    # snr = float(input("Enter the SNR (in dB): "))
    print(f"SNR: {snr}")
    for a in range(num_iterations):
        print(f"Iteration: {a + 1}")

        bit_sequence = [1, 0, 0, 0, 1, 0, 1, 1]

        navic_sequence = [complex(0, 0)] * (20000 * 2 * 8)

        doppler = [1502.36, 1567, 4501.2, 6768.69, 8901.45, 1567, 1596, 1100.2, 6709.1, 7891.2]
        # doppler = [4501.2, 4501.2, 4501.2, 4501.2, 4501.2 , 4501.2, 4501.2, 4501.2, 4501.2, 4501.2]
        doppler_estimate = [0] * 10
        for i in range(10):
            doppler_estimate[i] = doppler[i]-1.5
        timedelay = [196, 1789, 1545, 2123, 33, 745, 58, 569, 3734, 281]
        for i in range(1, 11):
            with open(f"Navic_PRN_{i}.txt", "r") as file:
                prn_init = list(map(int, file.read().strip().split()))
            # print(len(prn_init))
            prn = upsample(prn_init, 5 * 1e-3, 4 * 1e6)
            size = len(prn)
            # print(len(prn))
            prn.extend(prn[:size])

            prn = [-1 if bit == 1 else 1 for bit in prn]
            expanded_signal_navic = []

            for bit in bit_sequence:
                if bit == 1:
                    expanded_signal_navic.extend([-x for x in prn])
                else:
                    expanded_signal_navic.extend(prn)

            alpha = 1
            # Scale the signal by alpha
            expanded_signal_navic = [alpha * x for x in expanded_signal_navic]

            # Apply integer delay
            IntegerDelay_Navic(expanded_signal_navic, timedelay[i - 1], i)

            # Apply Doppler shift (sampling frequency = 4 MHz)
            expanded_signal_navic_complex = apply_doppler(expanded_signal_navic, doppler[i - 1], 4_000_000)

            # Add the signal to the navic sequence
            for j in range(len(navic_sequence)):
                navic_sequence[j] += expanded_signal_navic_complex[j]

            # print(len(expanded_signal_navic_complex))
        navic_sequence = add_gaussian_noise(navic_sequence, snr)
        # print(len(navic_sequence))

        gps_sequence = [0j] * (4000 * 10 * 8)

        for i in range(1, 11):
            prn_gps_init = PRN(i)
            prn_gps = upsample(prn_gps_init, 1e-3, 4e6)
            size = len(prn_gps)

            prn_gps *= 10

            prn_gps = [1 if x == 1 else -1 for x in prn_gps]

            expanded_signal_gps = []
            for bit in bit_sequence:
                if bit == 1:
                    expanded_signal_gps.extend(prn_gps)
                else:
                    expanded_signal_gps.extend([-x for x in prn_gps])

            # Scale by alpha
            alpha = 1
            expanded_signal_gps = [alpha * x for x in expanded_signal_gps]

            # Apply integer delay
            IntegerDelay_gps(expanded_signal_gps, timedelay[i - 1], i)

            # Apply doppler shift
            expanded_signal_gps_complex = apply_doppler(expanded_signal_gps, doppler[i - 1], 4_000_000)

            # Add signal to the gps sequence
            for j in range(len(gps_sequence)):
                gps_sequence[j] += expanded_signal_gps_complex[j]

        # Add noise
        gps_sequence = add_gaussian_noise(gps_sequence, snr)

        navic_doppler_estimates = []
        gps_doppler_estimates = []
        navic_code_phase_estimates = []
        gps_code_phase_estimates = []

        with open(f"Navic_PRN_{1}.txt", "r") as file:
            prn10_cpe_navic_init = list(map(int, file.read().strip().split()))
        prn10_cpe_navic = upsample(prn10_cpe_navic_init, 5e-3, 4e6)
        prn10_cpe_navic = [-1 if bit == 1 else 1 for bit in prn10_cpe_navic]
        prn10_cpe_navic_complex = [complex(x) for x in prn10_cpe_navic]
        threshold = calculate_threshold(navic_sequence, 4e6,doppler_estimate[0], prn10_cpe_navic_complex)
        print("Threshold:", threshold)
        for i in range(10):
            print(f"Satellite: {i + 1}")
            print(f"Actual Doppler: {doppler[i]}")
            print(f"Code Phase Delay: {timedelay[i]}")
            
            doppler_estimate_10 = [doppler_estimate[i]]
            
            with open(f"Navic_PRN_{i+1}.txt", "r") as file:
                prn10_cpe_navic_init = list(map(int, file.read().strip().split()))
            prn10_cpe_navic = upsample(prn10_cpe_navic_init, 5e-3, 4e6)
            
            prn10_cpe_gps_init = PRN(i + 1)
            prn10_cpe_gps = upsample(prn10_cpe_gps_init, 1e-3, 4e6)
            
            prn10_cpe_gps *= 5  # repeat 5 times to match navic size approx
            
            print("Navic Size:", len(prn10_cpe_navic))
            print("GPS Size:", len(prn10_cpe_gps))

            prn10_cpe_navic = [-1 if bit == 1 else 1 for bit in prn10_cpe_navic]
            prn10_cpe_gps = [1 if bit == 1 else -1 for bit in prn10_cpe_gps]

            prn10_cpe_navic_complex = [complex(x) for x in prn10_cpe_navic]
            prn10_cpe_gps_complex = [complex(x) for x in prn10_cpe_gps]

            print("Navic Estimate:")
            navic_estimates = code_phase_doppler_estimate(navic_sequence, 4e6, doppler_estimate_10, prn10_cpe_navic_complex,threshold)
            code_phase_estimate_navic = navic_estimates[0]
            doppler_estimate_navic = navic_estimates[1]

            print("GPS Estimate:")
            gps_estimates = code_phase_doppler_estimate(gps_sequence, 4e6, doppler_estimate_10, prn10_cpe_gps_complex,threshold)
            code_phase_estimate_gps = gps_estimates[0]
            doppler_estimate_gps = gps_estimates[1]

            navic_code_phase_estimates.append(code_phase_estimate_navic)
            gps_code_phase_estimates.append(code_phase_estimate_gps)
            navic_doppler_estimates.append(doppler_estimate_navic)
            gps_doppler_estimates.append(doppler_estimate_gps)
            print("\n")

        print(f"Estimation completed for iteration {a + 1}")
        print(len(navic_code_phase_estimates), num_iterations)

        for w in range(len(navic_code_phase_estimates)):
            average_code_phase_estimate_gps[w] += gps_code_phase_estimates[w] / num_iterations
            average_code_phase_estimate_navic[w] += navic_code_phase_estimates[w] / num_iterations
            average_doppler_estimate_gps[w] += gps_doppler_estimates[w] / num_iterations
            average_doppler_estimate_navic[w] += navic_doppler_estimates[w] / num_iterations
    write_estimates_to_file(
        snr,
        average_code_phase_estimate_gps,
        average_code_phase_estimate_navic,
        average_doppler_estimate_gps,
        average_doppler_estimate_navic
    )

