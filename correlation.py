############################### IGNORE THIS FILE ###############################




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpssignal import gpssignalgen

gpssignal = gpssignalgen(filepath = 'telemetry/mydata2.dat')
originaldata1 = np.fromfile('data_split/mydataa', dtype=np.uint8)
originaldata2 = np.fromfile('data_split/mydatab', dtype=np.uint8)

binary_array = np.concatenate([np.array(list(f"{x:08b}"), dtype=int) for x in originaldata1])
series1 = pd.Series(originaldata1)[:1000]
series2 = pd.Series(gpssignal)[:100]
rolling_corr = series1.rolling(window=len(series2)).corr(series2)

plt.figure(figsize=(10, 6))
plt.plot(rolling_corr, label='Rolling Correlation', color='blue')
plt.title('Running Correlation between Two Arrays')
plt.xlabel('Index')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.show()