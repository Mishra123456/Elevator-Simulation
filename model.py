import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, lr=0.01):
        super(QNetwork, self).__init__()
        # Input: Nfloor, Npeople, Nstop, action
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        x is a tensor of shape (batch, 4)
        Returns the Q-value Q(s, a).
        For cost minimization, lower Q is better.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def update(self, s, target_q):
        """
        Perform a single gradient descent step.
        """
        self.optimizer.zero_grad()
        
        # s is shape (batch, 4), target_q is shape (batch, 1)
        pred_q = self.forward(s)
        loss = self.criterion(pred_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
