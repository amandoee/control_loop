from ast import literal_eval
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import numpy as np

class LocalizationNet(nn.Module):
    def __init__(self):
        super(LocalizationNet, self).__init__()
        # LiDAR branch: Input is a 1D signal of length 1080
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, 1350)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # output: (batch, 16, 675)
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # output: (batch, 32, 270)
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # output: (batch, 64, 135)
        )
        # Fully connected part: flatten and regress to 4 outputs (x, y, qw, qz)
        self.fc = nn.Sequential(
            nn.Linear(10752, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.last_layer = nn.Sequential(
            nn.Linear(128,3)
        )
        
    def forward(self, lidar, angle):
        x = lidar.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        #y = angle.unsqueeze(1)
        #z = torch.cat((x,y), dim=-1)
        x = self.fc(x)
        out = self.last_layer(x)
        return out


import torch
import torch.nn as nn

class XYAngleLoss(nn.Module):
    def __init__(self, xy_weight=1.0, angle_weight=1.0):
        """
        Loss function for XY coordinates and angle in radians
        
        Args:
            xy_weight: Weight for the XY coordinates loss
            angle_weight: Weight for the angle loss
        """
        super(XYAngleLoss, self).__init__()
        self.xy_weight = xy_weight
        self.angle_weight = angle_weight
    
    def forward(self, pred, target):
        """
        Calculate the weighted loss for XY coordinates and angle
        
        Args:
            pred: Predicted values [batch_size, 3] where [:, 0:2] are x,y and [:, 2] is the angle in radians
            target: Target values [batch_size, 3] where [:, 0:2] are x,y and [:, 2] is the angle in radians
            
        Returns:
            Combined loss value
        """
        # Split the predictions and targets
        pred_xy = pred[:, 0:2]
        pred_angle = pred[:, 2]
        
        target_xy = target[:, 0:2]
        target_angle = target[:, 2]
        
        # Calculate XY loss (Mean Squared Error)
        xy_loss = torch.mean(torch.sum((pred_xy - target_xy) ** 2, dim=1))
        
        # Calculate angle loss using 1 - cos(angle_diff)
        # This handles the circular nature of angles
        angle_diff = pred_angle - target_angle
        angle_loss = torch.mean(1-torch.cos(angle_diff))
        
        # Combine the losses
        combined_loss = self.xy_weight * xy_loss + self.angle_weight * angle_loss
        
        return combined_loss



if __name__ == '__main__': 
    dataloaders = []
    for i in range(2):
        df = pd.DataFrame()
        if i == 0:
            df = pd.read_csv('lidar_data.csv', sep=';') #f'movement_data/movement_data{i}.csv', sep=";")
        else:
            df = pd.read_csv('movement_data/movement_data0.csv',sep=';')
        lidar_data = []
        targets= []
        for idx in range(len(df)-1):
            row1 = df.iloc[idx]
            row2 = df.iloc[idx+1]
            scan_str = row1['lidar_scan']
            scan_str2 = row2['lidar_scan']
            pos_x = row1['x'] - row2['x']
            pos_y = row1['y'] - row2['y']
        
            yaw = (row1['yaw']- row2['yaw'])

            # Extract the list of ranges from the string, e.g. "ranges=[...]" 
            ranges = np.array(literal_eval(f'{scan_str}'))
            ranges2 = np.array(literal_eval(f'{scan_str2}'))
            d_ranges = ranges - ranges2
            # print("test", pos_x,pos_y)
            full_range = [0*i for i in range(1350)]
            for index, i in enumerate(range(round(row1['yaw']/0.0043633),round(1080+row1['yaw']/0.0043633))):
                if index >= 1080:
                    break
                full_range[i % 1350] = d_ranges[index] 
            
            full_range = np.append(full_range, row1['yaw'])
            lidar_data.append(full_range)
            targets.append([pos_x,pos_y,yaw])
    
        # Extract target pose columns.
        lidar_data = lidar_data
        # Convert lists/dataframes to torch tensors.
        lidar_data = torch.tensor(lidar_data, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
    
        dataset = TensorDataset(lidar_data, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        dataloaders.append(dataloader)

    # Instantiate model, loss function, and optimizer.
    model = LocalizationNet()
    #model.load_state_dict(torch.load('localization_model.pth'))
    criterion = XYAngleLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #model.load_state_dict(torch.load("localization_model.pth", map_location=torch.device('cpu')))
    # Training loop on CPU.
    num_epochs = 600
    model.train()

    best_val_loss = 1000
    for epoch in range(num_epochs):
        running_loss = 0.0
        validation_loss = 0.0
        total_length = 0
        validation_length = 1
        for map_idx, dataloader in enumerate(dataloaders):
            for batch_idx, (lidar_batch, target_batch) in enumerate(dataloader):    
                optimizer.zero_grad()
                print(torch._shape_as_tensor(lidar_batch))
                outputs = model(lidar_batch[:,:1350],lidar_batch[:,1350])
                loss = criterion(outputs, target_batch)
                if map_idx == 1: #batch_idx >= len(dataloader)*0.75:
                    validation_loss += loss.item()
                    validation_length = len(dataloader)
                    continue
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_length = len(dataloader)
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total_length:.4f} Validation loss: {validation_loss/validation_length:.4f}", end ="")

        if validation_loss/validation_length < best_val_loss:
            best_val_loss = validation_loss/validation_length 
            print("\nbest validation loss: ", best_val_loss)
            torch.save(model.state_dict(), 'localization_model.pth')
                #os.system("python test_NN.py")
        

    #torch.save(model.state_dict(), 'localization_model.pth')





    
