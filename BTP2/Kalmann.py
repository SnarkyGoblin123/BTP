import numpy as np
import matplotlib.pyplot as plt
from Extract_doppler import read_doppler_data

class KalmanFilter:
    def __init__(self):
        self.state, self.P, self.F, self.Q, self.H, self.R = self.initialize_kalman_filter()

    def initialize_kalman_filter(self):
        state = np.array([0, 0])  # Initial state: [Doppler, Rate of Change]
        P = np.eye(2) * 1e3       # Initial state covariance
        F = np.array([[1, 0.01],  # State transition model
                      [0, 1]])
        Q = np.array([[0.1, 0],   # Process noise covariance
                      [0, 0.1]])
        H = np.array([[1, 0]])    # Observation model
        R = np.array([[10]])      # Measurement noise covariance
        return state, P, F, Q, H, R

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, z):
        y = z - (self.H @ self.state)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(len(self.state)) - K @ self.H) @ self.P
        return self.state

    def run_filter(self, measurements):
        estimated_doppler = []
        # for z in measurements:
        self.predict()
        self.update(measurements)
        estimated_doppler.append(self.state[0])
        return estimated_doppler

    def predict_future(self, steps):
        future_doppler = []
        for _ in range(steps):
            self.predict()
            future_doppler.append(self.state[0])
        return future_doppler

filename = '/home/joel/BTP/./GSDR028t57.25O'
real_doppler, sat, time = read_doppler_data(filename)
real_doppler = real_doppler[7]

# Initialize Kalman filter
kf = KalmanFilter()
index = [i for i in range(500)]

index.extend([500 + 0*i for i in range(1000)])
estimated_doppler = []
# print(index)
for i in range(1500):
    if i in index:
        estimated_doppler.append(kf.run_filter(real_doppler[i]))
    else:
        estimated_doppler.append(kf.predict_future(1))
# # Run Kalman filter on real data
# estimated_doppler = kf.run_filter(real_doppler[:1000])

# # Predict Doppler for the next 10 seconds
# future_doppler = kf.predict_future(500)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, real_doppler, label="Real Doppler Data", color="blue", linewidth=2)
plt.plot(time[:1500], estimated_doppler, label="Estimated Doppler", color="orange", linestyle="--")
# plt.plot(time[1000] + np.arange(1, 501), future_doppler, label="Predicted Doppler (Next 10s)", color="green", linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Doppler Shift (Hz)")
plt.title("Doppler Shift Prediction with Kalman Filter")
plt.legend()
plt.grid()
plt.show()
