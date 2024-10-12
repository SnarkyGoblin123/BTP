import numpy as np
import goldcodegenerator as gcg
import struct
import copy
import sys

k = float(sys.argv[1])

filepath = '/home/joel/gps-sdr-sim-master/gen_data_split/mydataa'
data = np.fromfile(filepath,dtype = np.short)
compdata = data[::2] + (1j)*data[1::2]

sv = 10
prn = gcg.PRN(sv)
prn = [-1 if x == 0 else x for x in prn]

prn_upsample = []
n = 0
# Upsamples the PRN code to 4MHz
for i in range(4000):
    if ((i+1)/4000 <= (n+1)/1023):
        prn_upsample.append(prn[n])
    else:
        n = n+1
        prn_upsample.append(prn[n])

doppler = {18 : 0, 12: 2697.905, 13: -2848.802, 10: 2891.328, 5: -2449.682, 15: -1816.517, 23: 2137.817}

gold_code = []
# Repeats the PRN code 20 times
for i in range(20):
  gold_code.append(prn_upsample)
gold_code = np.array(gold_code).flatten()

code = [1,0,0,0,1,0,1,1] # Initial 8 bits of the processed I data

# Generate the input vector from gold code and I values
v1 = []
for i in range(len(code)):
  if code[i] == 1:
    v1.append(np.array(gold_code))
  else:
    v1.append(np.multiply(-1,np.array(gold_code)))
v1 = np.array(v1).flatten()

# Shift the complex data by the Doppler frequency
f_d = doppler[sv]
n = np.arange(len(compdata))
compdata_shifted = np.multiply(compdata,np.exp(-1j*2*np.pi*f_d*n/4000000))

# After performing the correlation in test.ipynb, we take the first maxima which comes out to be at 3200
y = copy.deepcopy(compdata_shifted[120323923: 120323923 + 640000])
x = v1

# Least squares estimation for the eqn y = alp*x + noise
alp_est = np.dot(x.T, y)/np.dot(x.T,x)
alp_est = k * alp_est

# Subtract the estimated alp*x from y to remove satellite 23 component for atleast those 640000 samples
yhat = y - alp_est*x


# Perform inverse doppler shift to get the original signal
yhat = np.multiply(yhat, np.exp(1j*2*np.pi*f_d*n[120323923: 120323923 + 640000]/4000000))
# print (np.array_equal(yhat, compdata[3200: 3200 + 640000]))
# Update the complex data with the new data
compdata_upd = copy.deepcopy(compdata)

del compdata
del n
del compdata_shifted

compdata_upd[120323923: 120323923 + 640000] = yhat


# print (np.array_equal(compdata_upd, compdata))
compdata_upd = np.round(compdata_upd)
# print (np.array_equal(compdata_upd, compdata))
# Save the updated data
# Separate the real (I) and imaginary (Q) parts
I_data = np.real(compdata_upd).astype(np.short)
Q_data = np.imag(compdata_upd).astype(np.short)



# print (np.array_equal(I_data, data[::2]))
# print (np.array_equal(Q_data, data[1::2]))
# print(np.allclose(I_data, data[::2]))
# print(np.allclose(Q_data, data[1::2]))

# Interleave I and Q data
interleaved_data = np.empty((I_data.size + Q_data.size,), dtype=np.short)
interleaved_data[::2] = I_data
interleaved_data[1::2] = Q_data

# print (np.array_equal(interleaved_data, data))
# for i in range(200):
#   print (i)
#   print (interleaved_data[i], data[i])

      

# Pack the interleaved data into binary format
# packed_data = struct.pack(f'<{len(interleaved_data)}h', *interleaved_data)
packed_data = interleaved_data.tobytes()

# Write the packed data to a .dat file
output_filepath = '/home/joel/gps-sdr-sim-master/gen_data_split/output_upd.dat'
with open(output_filepath, 'wb') as f:
    f.write(packed_data)
    b1 = bytearray(b'_')
    f.write(b1)

print(f"Packed IQ data written to {output_filepath}")

