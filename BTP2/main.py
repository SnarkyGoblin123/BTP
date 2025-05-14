import numpy as np
from scipy.signal import fftconvolve
import os
import matplotlib.pyplot as plt
from Extract_doppler import read_doppler_data
import copy

class GNSSProcessor:
    def __init__(self, gnss_filepath, rinex_observation_filepath, telemetry_datapath, satellite_id, sample_rate=4000000):
        self.gnss_filepath = gnss_filepath
        self.rinex_observation_filepath = rinex_observation_filepath
        self.telemetry_datapath = telemetry_datapath
        self.satellite_id = satellite_id
        self.sample_rate = sample_rate
        self.doppler_data, self.satellites, self.times = read_doppler_data(rinex_observation_filepath)
        if satellite_id not in self.satellites:
            raise ValueError(f"Satellite {satellite_id} not found in Observation data.")
        self.satellite_index = self.satellites.index(satellite_id)
        self.doppler_values = self.doppler_data[satellite_id]
        print(self.doppler_values[0])
        self.gold_code = self.generate_gold_code(satellite_id, int(sample_rate/1000))
        self.code_phase_est = 0
        self.init_doppler_estimate = 0
        self.telemdata = np.fromfile(self.telemetry_datapath, dtype = np.short)
        self.extract_telem_bits()
        self.GPS_signal = self.generate_spread_signal(self.telem_bits)
        self.spread_preamble = self.GPS_signal[:640000]    

    def code_phase_estimate(self, data):
        max_corr = 0
        best_index = 0
        best_doppler = 0
        doppler = self.doppler_values[0]
        print(doppler)
        comp_corr=0
        for doppler_shift in np.arange(-10, 10, 1):
            shifted_data = self.apply_doppler_shift(data, doppler, self.sample_rate, doppler_shift, 70000000)
            abs_corr = self.correlate(shifted_data, self.spread_preamble)
            max_abs_corr = np.max(np.abs(abs_corr))
            if max_abs_corr > max_corr:
                max_corr = max_abs_corr
                best_index = np.argmax(np.abs(abs_corr))
                best_doppler = doppler_shift
                comp_corr = abs_corr[best_index]
        self.code_phase_est = best_index
        self.init_doppler_estimate = best_doppler
        print(self.code_phase_est)
        print(self.init_doppler_estimate)
        print(max_corr)
        print(comp_corr)

    def fine_doppler_shift(self, compdata, st, chunk, fd, fs, search_param = [-10, 1, 10], imag_threshold=200000):
        best_shift = 10000
        max_correlation = 0
        prev_min = 1e7
        curr_min = 1e7
        best_corr_arr = None
        for doppler_shift in [search_param[0]+search_param[1]*i for i in range(int((search_param[2]-search_param[0])/search_param[1]))]:
            shifted_data = self.apply_doppler_shift(compdata, fd, fs, doppler_shift, st)
            corr = self.correlate(shifted_data, chunk)
            real_corr = np.real(corr)
            imag_corr = np.imag(corr)
            
            if np.all(np.abs(imag_corr) < imag_threshold):
                # print(imag_corr[:20])
                curr_min = np.mean(np.abs(imag_corr)) #Finding avg in imag_corr and checking if it is less than prev avg
                if (curr_min < prev_min):
                    prev_min = curr_min
                    best_shift = doppler_shift
                    best_corr_arr = copy.deepcopy(corr)
                    
                # break
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.real(best_corr_arr), label='Real Part', color='blue')
        plt.plot(np.imag(best_corr_arr), label='Imaginary Part', color='red')
        plt.title(f'Fine Doppler Correction: {best_shift}')
        plt.xlabel('Sample Index')
        plt.ylabel('Correlation Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        return best_shift, best_corr_arr

    def PRN(self, sv):
        """Build the CA code (PRN) for a given satellite ID"""
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

    def apply_doppler_shift(self, compdata, fd, fs, doppler_shift, st):
        n = np.arange(len(compdata)) + st
        shift_factor = np.exp(-1j * 2 * np.pi * (fd + doppler_shift) * n / fs)
        return compdata * shift_factor

    def correlate(self, compdata, chunk):
        correlation = fftconvolve(compdata, chunk[::-1], mode='valid')
        return correlation

    def generate_gold_code(self, sv, upsample_rate=4000):
        """Generates upsampled gold code for a given satellite."""
        prn = self.PRN(sv)
        prn = [-1 if x == 0 else x for x in prn]
        prn_upsample = []
        n = 0
        for i in range(upsample_rate):
            if ((i+1)/upsample_rate <= (n+1)/1023):
                prn_upsample.append(prn[n])
            else:
                n = n+1
                prn_upsample.append(prn[n])
        return np.array(prn_upsample)
    
    def extract_telem_bits(self):
        self.telem_bits = self.telemdata[12::16]
        self.telem_bits = self.telem_bits*(-1*self.telem_bits[0])
        print(len(self.telem_bits))
        self.telem_bits = self.telem_bits[:5000]
        

    def generate_spread_signal(self, data):
        spread_telem = np.repeat(data[1:], 20*(len(self.gold_code)))
        sat_tiled = np.tile(self.gold_code, 20*len(data[1:]))
        final= spread_telem * sat_tiled
        return final


    def subtract_satellite_signal(self, output_filename="output/output.dat"): 
        """
        Reads GNSS data in chunks, subtracts the spreaded telemetry signal using linear regression, and saves the processed data.
        """
        p = int(self.sample_rate/1000) * 20 # Number of samples per telemetry bit
        # st = self.code_phase_est # Start index
        st = 72324011
        chunk_size = int(20*self.sample_rate/1000 * 4)
        skip = st // p
        skip = int(skip)
        print(skip)
        print(len(self.GPS_signal))
        self.doppler_values[0] = 2891.328
        # Prepare output file
        with open(output_filename, "wb") as outfile:
            # Open and process the GNSS data in chunks

            with open(self.gnss_filepath, "rb") as infile:
                chunk_num = 0
                file_offset = 0  # Keep track of the file offset in samplea

                # Skip until we reach st
                for i in range(skip):
                    chunk = infile.read(chunk_size)
                    data = np.frombuffer(chunk, dtype = np.short)
                    # Save the processed chunk
                    data.tofile(outfile)
                    print(f"Processed chunk {chunk_num}, file_offset {file_offset}")
                    chunk_num += 1
                    file_offset += int(len(data)/2)  # Update the file offset
                
                last_file_offset = file_offset
                while True:
                    if chunk_num == skip:
                        chunk_size_ = chunk_size + int((st%chunk_size)*4)
                        print(chunk_size_/2)
                    else:
                        chunk_size_ = chunk_size
                    chunk = infile.read(chunk_size_)
                    if not chunk:
                        print("Exiting due to no chunk")
                        break

                    # Convert chunk to complex data
                    data = np.frombuffer(chunk, dtype=np.short)
                    x = data[::2] + 1j*data[1::2]

                    # Apply Doppler shift
                    chunk_start_time = file_offset / self.sample_rate  # Time in seconds
                    doppler_index = np.argmin(np.abs(np.array(self.times) - chunk_start_time))
                    if (doppler_index == 0):
                        doppler_shift = self.doppler_values[doppler_index]
                    else:
                        doppler_shift = self.doppler_values[doppler_index]
 
                    n = file_offset
                    

                    # Calculate the actual start index within the current chunk
                    actual_start = max(0, st - file_offset)

                    # Perform subtraction with linear regression
                    num_iterations = min(len(x) - actual_start, len(self.GPS_signal) - (chunk_num*80000-last_file_offset)) // p
                    num_iterations = max(num_iterations, 0)
                    if num_iterations != 0:
                        x_shifted = self.apply_doppler_shift(x, doppler_shift, self.sample_rate, 0, n)

                        # print("before sub", num_iterations)
                        for i in range(num_iterations):
                            start_index = actual_start + i * p
                            end_index = start_index + p

                            # Extract segments
                            x_segment = x_shifted[start_index:end_index]
                            gps_segment = self.GPS_signal[chunk_num*80000 - last_file_offset:chunk_num*80000 + p - last_file_offset]

                            # Linear Regression to find amplitude 'a'
                            alpha = np.dot(gps_segment.T, x_segment)/np.dot(gps_segment.T,gps_segment) #1.639 -93.247j
                            
                            # Subtract the signal
                            x_shifted[start_index:end_index] -= 1.15*alpha * gps_segment

                        # Perform inverse doppler shift to get the original signal

                        x_inverse_shifted = self.apply_doppler_shift(x_shifted, -doppler_shift, self.sample_rate, 0, n)

                        # Convert back to int16 for saving (real and imag parts separately)
                        corrected_data = np.empty(len(x_inverse_shifted) * 2, dtype=np.short)
                        corrected_data[::2] = np.real(x_inverse_shifted).astype(np.short)
                        corrected_data[1::2] = np.imag(x_inverse_shifted).astype(np.short)
                        corrected_data.tofile(outfile)
                    else:
                        data.tofile(outfile)
                    
                    print(f"Processed chunk {chunk_num}, file_offset {file_offset}")

                    chunk_num += 1
                    file_offset += len(x)  # Update the file offset

# Example Usage
# gnss_data_path = '/home/joel/data/split/mydataa'  # Replace with your GNSS data file
# code_data_path = '/home/joel/gps-sdr-sim-master/gen_data_split/mydataa'
code_data_path = '/home/joel/gps-sdr-sim-master/prn_gain0.75/split_data/mydataa' 
gnss_data_path = '/home/joel/gps-sdr-sim-master/gpsgen_05092024.dat'
doppler_data_path = '/home/joel/BTP/GSDR050t25.25O'  # Replace with your doppler data file'
telemetry_data_path = '/home/joel/BTP/telemetry/mydata1.dat'
satellite_to_subtract = 10

data = np.fromfile(code_data_path, dtype = np.short)
print(len(data))
compdata = data[0::2] + 1j*data[1::2]
st = 000000
fd = 2891.328
compdata = compdata[st:st + 1200000]
processor = GNSSProcessor(gnss_data_path, doppler_data_path, telemetry_data_path, satellite_to_subtract)
# fd = 2891.328
# fine_doppler for st = 7e7 -1.09
# fine_doppker for st = 0, -2.21
# fine doppler for st = 1e7 = -2.01
fine_dopp_correction, _ = processor.fine_doppler_shift(compdata, st, processor.spread_preamble[:4000], fd, processor.sample_rate, [-2.5,0.01,-2])
# fine_dopp_correction = -2.3
shifted_data = processor.apply_doppler_shift(compdata, fd + fine_dopp_correction , processor.sample_rate, 0, st)
corr = processor.correlate(shifted_data, processor.spread_preamble)
del compdata
del shifted_data
del data
print(np.argmax(np.abs(corr)))
plt.figure(figsize=(10, 6))
plt.plot(np.real(corr), label='Real Part', color='blue')
plt.plot(np.imag(corr), label='Imaginary Part', color='red')
plt.title('Correlation Output')
plt.xlabel('Sample Index')
plt.ylabel('Correlation Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# shift, corr = processor.fine_doppler_shift(compdata, 70000000, processor.spread_preamble[:4000], processor.init_doppler_estimate-3, processor.sample_rate, [-5,0.01,5])
# print((np.where(abs(np.real(corr)) > 250000)[0]))
# print(shift)
# processor.subtract_satellite_signal()

# processor.subtract_satellite_signal()
