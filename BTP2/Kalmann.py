import numpy as np
"""
Kalmann.py

This script implements a Kalman Filter for estimating and predicting Doppler shift data. 
The Kalman Filter is initialized with predefined parameters and processes real Doppler 
data obtained from a file. The script also visualizes the real Doppler data alongside 
the estimated and predicted Doppler data.

This can be utilized for spoofing this way. Say we have the past correct doppler, we can predict the doppler for the next 5 seconds or so
Use this to generate the data for spoofing of th enext 5 seconds beforehand and at theexact time send the signal for spoofing

Classes:
    KalmanFilter: A class that encapsulates the Kalman Filter algorithm, including 
                  initialization, prediction, update, and future prediction.

Functions:
    read_doppler_data(filename): Reads Doppler data from the specified file. 
                                 (Imported from Extract_doppler module)

Usage:
    - The KalmanFilter class is used to estimate Doppler shift data from measurements.
    - The `run_filter` method processes real Doppler data to estimate the Doppler shift.
    - The `predict_future` method predicts future Doppler shifts for a specified number of steps.
    - The script visualizes the real Doppler data, estimated Doppler data, and optionally 
      predicted future Doppler data using matplotlib.

Attributes:
    state (numpy.ndarray): The current state vector of the Kalman Filter.
    P (numpy.ndarray): The state covariance matrix.
    F (numpy.ndarray): The state transition model.
    Q (numpy.ndarray): The process noise covariance matrix.
    H (numpy.ndarray): The observation model.
    R (numpy.ndarray): The measurement noise covariance matrix.

Methods:
    initialize_kalman_filter():
        Initializes the Kalman Filter parameters including state, covariance, 
        transition model, process noise, observation model, and measurement noise.

    predict():
        Performs the prediction step of the Kalman Filter, updating the state 
        and covariance based on the state transition model.

    update(z):
        Performs the update step of the Kalman Filter using the provided measurement `z`.
        Updates the state and covariance based on the Kalman gain and innovation.

    run_filter(measurements):
        Runs the Kalman Filter on a sequence of measurements to estimate the Doppler shift.
        Args:
            measurements (list or numpy.ndarray): The sequence of Doppler measurements.
        Returns:
            list: The estimated Doppler values.

    predict_future(steps):
        Predicts future Doppler shifts for a specified number of steps using the Kalman Filter.
        Args:
            steps (int): The number of future steps to predict.
        Returns:
            list: The predicted Doppler values.

Example:


    # Run Kalman filter on real data
    estimated_doppler = kf.run_filter(real_doppler[:1000])

    # Predict Doppler for the next 10 seconds
    future_doppler = kf.predict_future(500)

"""
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
