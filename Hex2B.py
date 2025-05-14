import numpy as np

# Manan garg ke tassali ke liye

data = np.fromfile('data_split/mydataa', dtype=np.uint8)
bin_repr = np.unpackbits(data)

# with open('data_split/binary.bin', 'wb') as f:
#     f.write(bin_repr)
#     f.close()
    # print(bin_repr[:16])

    # data = np.frombuffer(data, dtype=np.uint8)
    # data = data.bin()
    # print(data)