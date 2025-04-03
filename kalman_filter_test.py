import numpy as np

def generate_data(A, H, Q, R, x0, T):
    """
    Generate synthetic data for a linear state-space model.
    
    Parameters:
      A  : state transition matrix.
      H  : observation matrix.
      Q  : process noise covariance.
      R  : measurement noise covariance.
      x0 : initial state (column vector).
      T  : number of time steps.
      
    Returns:
      x: list of true state column vectors.
      z: list of noisy observation column vectors.
    """
    n = A.shape[0]
    m = H.shape[0]
    x = [None] * T
    z = [None] * T
    x[0] = x0
    for t in range(1, T):
        # Process noise
        w = np.random.multivariate_normal(np.zeros(n), Q).reshape(n, 1)
        x[t] = A @ x[t-1] + w
    for t in range(T):
        # Measurement noise
        v = np.random.multivariate_normal(np.zeros(m), R).reshape(m, 1)
        z[t] = H @ x[t] + v
    return x, z

def kalman_filter(z, A, H, Q, R, x0, P0):
    """
    Run a forward Kalman filter.
    
    Returns:
      x_filt: list of filtered state estimates.
      P_filt: list of corresponding state covariance matrices.
      x_pred: list of predicted state estimates.
      P_pred: list of predicted state covariance matrices.
    """
    T = len(z)
    n = A.shape[0]
    x_filt = [None] * T
    P_filt = [None] * T
    x_pred = [None] * T
    P_pred = [None] * T

    # Initialize with the given state and covariance.
    x_filt[0] = x0
    P_filt[0] = P0
    x_pred[0] = x0
    P_pred[0] = P0

    for t in range(1, T):
        # Prediction step
        x_pred[t] = A @ x_filt[t-1]
        P_pred[t] = A @ P_filt[t-1] @ A.T + Q
        
        # Update step
        S = H @ P_pred[t] @ H.T + R  # Innovation covariance
        K = P_pred[t] @ H.T @ np.linalg.inv(S)  # Kalman gain
        y = z[t] - H @ x_pred[t]  # Innovation
        x_filt[t] = x_pred[t] + K @ y
        P_filt[t] = (np.eye(n) - K @ H) @ P_pred[t]
        
    return x_filt, P_filt, x_pred, P_pred

def kalman_smoother(z, A, H, Q, R, x0, P0):
    """
    Rauch–Tung–Striebel (RTS) smoother.
    
    Returns:
      x_smooth: list of smoothed state estimates.
      P_smooth: list of smoothed state covariances.
      P_lag   : list of lag-one covariances (for t >= 1, P_lag[t] = Cov(x_t, x_{t-1} | z_{1:T})).
    """
    T = len(z)
    # Run the forward Kalman filter.
    x_filt, P_filt, x_pred, P_pred = kalman_filter(z, A, H, Q, R, x0, P0)
    
    # Initialize arrays for smoothed estimates.
    x_smooth = [None] * T
    P_smooth = [None] * T
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # Compute the smoother gains.
    J = [None] * (T - 1)
    for t in range(T - 1):
        J[t] = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t+1])
    
    # Backward pass to compute smoothed estimates.
    for t in reversed(range(T - 1)):
        x_smooth[t] = x_filt[t] + J[t] @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + J[t] @ (P_smooth[t+1] - P_pred[t+1]) @ J[t].T

    # Compute lag-one covariances.
    P_lag = [None] * T
    P_lag[0] = None  # Undefined for t=0.
    for t in range(1, T):
        P_lag[t] = J[t-1] @ P_smooth[t]
    
    return x_smooth, P_smooth, P_lag

def EM_Kalman(z, A, H, Q_init, R_init, x0, P0, num_iters=10):
    """
    EM algorithm for learning Q and R using only the noisy observations z.
    
    In the E-step, we run the Kalman smoother to obtain the smoothed state estimates
    and covariances. In the M-step, we update Q and R using these estimates.
    
    Parameters:
      z       : list of observations (each a column vector).
      A, H    : known state transition and observation matrices.
      Q_init  : initial guess for the process noise covariance.
      R_init  : initial guess for the measurement noise covariance.
      x0, P0  : initial state and its covariance.
      num_iters: number of EM iterations.
      
    Returns:
      Q, R    : learned noise covariances.
    """
    T = len(z)
    Q = Q_init.copy()
    R = R_init.copy()
    
    for iteration in range(num_iters):
        # E-Step: Run Kalman smoother to estimate the latent states.
        x_smooth, P_smooth, P_lag = kalman_smoother(z, A, H, Q, R, x0, P0)
        
        # M-Step: Update Q and R.
        Q_new = np.zeros_like(Q)
        for t in range(1, T):
            # Compute expected outer product of the state at time t.
            Exx = P_smooth[t] + x_smooth[t] @ x_smooth[t].T
            # Predicted mean based on previous state.
            pred_mean = A @ x_smooth[t-1]
            # Update using the identity:
            # E[(x_t - A*x_{t-1})(x_t - A*x_{t-1})^T] =
            #   E[x_t x_t^T] - A*E[x_{t-1} x_t^T] - E[x_t x_{t-1}^T]*A^T + A*E[x_{t-1} x_{t-1}^T]*A^T
            Q_new += (Exx 
                      - pred_mean @ x_smooth[t].T 
                      - x_smooth[t] @ pred_mean.T 
                      + pred_mean @ pred_mean.T)
        Q_new /= (T - 1)
        
        R_new = np.zeros_like(R)
        for t in range(T):
            residual = z[t] - H @ x_smooth[t]
            R_new += residual @ residual.T + H @ P_smooth[t] @ H.T
        R_new /= T
        
        Q, R = Q_new, R_new
        print(f"Iteration {iteration+1}:\nQ =\n{Q}\nR =\n{R}\n")
        
    return Q, R

def main():
    np.random.seed(0)
    
    # Define system matrices for a 1D constant velocity model.
    dt = 1.0
    A = np.array([[1, dt],
                  [0,  1]])
    H = np.array([[1, 0]])  # We only observe the position.
    
    # True noise covariances (unknown in practice)
    Q_true = np.array([[0.1, 0],
                       [0, 0.1]])
    R_true = np.array([[0.5]])
    
    # Initial state and covariance.
    x0 = np.array([[0],
                   [1]])  # Starting at position 0 with velocity 1.
    P0 = np.eye(2)
    
    T = 100 # Number of time steps.
    
    # Generate synthetic data (x is generated internally but only z is used in EM).
    x_true, z = generate_data(A, H, Q_true, R_true, x0, T)
    
    # Initial guesses for Q and R.
    Q_init = np.eye(2)
    R_init = np.eye(1)
    
    # Run the EM algorithm using only the noisy observations z.
    Q_est, R_est = EM_Kalman(z, A, H, Q_init, R_init, x0, P0, num_iters=10)
    
    print("Final Estimated Q:\n", Q_est)
    print("Final Estimated R:\n", R_est)

if __name__ == '__main__':
    main()
