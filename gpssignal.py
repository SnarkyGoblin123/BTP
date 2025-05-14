import numpy as np
import matplotlib.pyplot as plt
from goldcodegenerator import PRN

def gpssignalgen(filepath):
    chkdat = np.fromfile(filepath,dtype = np.short)
    sv = chkdat[14]
    chkdat = chkdat[12::16]
    chkdat = chkdat*(-1*chkdat[0])
    
    sat_20 = [-1 if x==0 else x for x in PRN(sv)]
    gpssignal = []
    print(len(sat_20))
    print(len(chkdat[1:]))
    i = 0
    # for item in chkdat[1:]:
    #     for j in range(20):
    #         for i in sat_20:
    #             gpssignal.append(item*i)
    # print(len(gpssignal))
    gpssignal1 = []
    chkdat_repeated = np.repeat(chkdat[1:], 20*(len(sat_20)))
    sat_20_tiled = np.tile(sat_20, 20*len(chkdat[1:]))
    gpssignal1 = chkdat_repeated * sat_20_tiled
    return gpssignal1
 
# print(len(gpssignal1))
# if np.array_equal(gpssignal,gpssignal1):
#     print('True')
# np.savetxt('sat_20_data.txt',gpssignal1)