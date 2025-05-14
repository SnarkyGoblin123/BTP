import matplotlib.pyplot as plt

def read_doppler_data(filename):
    with open(filename, 'r') as file:
        for line in file:
            if "END OF HEADER" in line:
                break
        satellites = []
        len_sat = 0
        times = []
        lines = file.readlines()
        doppler = [[] for _ in range(32)]
        counter = 0 # to know which satellite data is being read
        for line in lines:
            line_arr = line.split()
            if len(line_arr) == 8:
                counter = 0
                info = line.split()
                time = float(info[4])*60 + float(info[5])
                times.append(time)
                sv = info[-1].split('G')[1:]
                sv = [int(x) for x in sv]
                len_sat = len(sv)
                if not satellites:
                    for s in sv:
                        satellites.append(s)
                # print(satellites)
            else:
                dpp = float(line.split()[4])
                doppler[satellites[counter]].append(dpp)
                counter+=1
                
        # print(doppler)
    return doppler, satellites, times





# filename = '../GSDR028t57.25O'
# filename = '/home/joel/BTP/GSDR320w53.24O'
# doppler, sat, times = read_doppler_data(filename)

# import matplotlib.pyplot as plt

# satellite_id = 10

# if satellite_id in sat:
#     sat_index = sat.index(satellite_id)
#     plt.plot(times, doppler[satellite_id], label=f'Satellite {satellite_id}')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Doppler Shift')
#     plt.title(f'Doppler Shift vs Time for Satellite {satellite_id}')
#     plt.legend()
#     plt.show()
# else:
    # print(f'Satellite {satellite_id} not found in the data.')

