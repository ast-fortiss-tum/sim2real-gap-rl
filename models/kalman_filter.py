# kalman_filter.py
import torch
import torch.nn as nn

class LearnableKalmanFilter(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # State transition and control matrices (B used if you have a control input; else can remain zero)
        self.A = nn.Parameter(torch.eye(state_dim))
        self.B = nn.Parameter(torch.eye(state_dim, state_dim))
        # Observation matrix
        self.H = nn.Parameter(torch.eye(state_dim))
        # Log-diagonals for process noise covariance Q and observation noise covariance R.
        # We parameterize the diagonal elements in log-space to ensure positivity.
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim))
        self.log_R_diag = nn.Parameter(torch.zeros(state_dim))
    
    def get_Q(self):
        return torch.diag(torch.exp(self.log_Q_diag))
    
    def get_R(self):
        return torch.diag(torch.exp(self.log_R_diag))
    
    def forward(self, x_prev, P_prev, z, u=None):
        """
        x_prev: previous state estimate (tensor of shape [state_dim])
        P_prev: previous covariance matrix (tensor of shape [state_dim, state_dim])
        z: current noisy observation (tensor of shape [state_dim])
        u: control input (if any; defaults to zero)
        """
        if u is None:
            u = torch.zeros(self.state_dim, device=x_prev.device)
        # Prediction step
        x_pred = torch.matmul(self.A, x_prev) + torch.matmul(self.B, u)
        P_pred = torch.matmul(self.A, torch.matmul(P_prev, self.A.t())) + self.get_Q()
        # Kalman gain
        S = torch.matmul(self.H, torch.matmul(P_pred, self.H.t())) + self.get_R()
        K = torch.matmul(P_pred, torch.matmul(self.H.t(), torch.inverse(S)))
        # Update step
        innovation = z - torch.matmul(self.H, x_pred)
        x_new = x_pred + torch.matmul(K, innovation)
        I = torch.eye(self.state_dim, device=x_prev.device)
        P_new = torch.matmul(I - torch.matmul(K, self.H), P_pred)
        return x_new, P_new

class FixedKalmanFilter(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # Register buffers for the fixed Kalman filter matrices.
        self.register_buffer("A", torch.eye(state_dim))
        self.register_buffer("B", torch.eye(state_dim, state_dim))
        self.register_buffer("H", torch.eye(state_dim))
        # Fixed noise covariances are parameterized via their log-diagonals.
        self.register_buffer("log_Q_diag", torch.zeros(state_dim))
        self.register_buffer("log_R_diag", torch.zeros(state_dim))
    
    def get_Q(self):
        return torch.diag(torch.exp(self.log_Q_diag))
    
    def get_R(self):
        return torch.diag(torch.exp(self.log_R_diag))
    
    def forward(self, x_prev, P_prev, z, u=None):
        """
        Performs one fixed Kalman filter update.
        
        Args:
            x_prev: previous state estimate (tensor of shape [state_dim])
            P_prev: previous covariance matrix (tensor of shape [state_dim, state_dim])
            z: current noisy observation (tensor of shape [state_dim])
            u: control input (defaults to zero vector)
        
        Returns:
            x_new: updated state estimate
            P_new: updated covariance matrix
        """
        if u is None:
            u = torch.zeros(self.state_dim, device=x_prev.device)
        # Prediction step.
        x_pred = torch.matmul(self.A, x_prev) + torch.matmul(self.B, u)
        P_pred = torch.matmul(self.A, torch.matmul(P_prev, self.A.t())) + self.get_Q()
        # Compute Kalman gain.
        S = torch.matmul(self.H, torch.matmul(P_pred, self.H.t())) + self.get_R()
        K = torch.matmul(P_pred, torch.matmul(self.H.t(), torch.inverse(S)))
        # Update step.
        innovation = z - torch.matmul(self.H, x_pred)
        x_new = x_pred + torch.matmul(K, innovation)
        I = torch.eye(self.state_dim, device=x_prev.device)
        P_new = torch.matmul(I - torch.matmul(K, self.H), P_pred)
        return x_new, P_new

