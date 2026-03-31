


#-------------------------------------< Imports >-----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F







#-----------------------------------------------------------< Functions >----------------------------------------------------------

class ThreeDCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ThreeDCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

#----------------------------------------------------------------------------------
class CNNRNN(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, num_layers=2):
        super(CNNRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        x = x.view(batch_size * num_frames, C, H, W)
        
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        last_output = lstm_out[:, -1, :]
        
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output

#-----------------------------------------------------------------------------------
class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.3):
        super(VideoTransformer, self).__init__()
        
        self.d_model = d_model
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        x = x.view(batch_size * num_frames, C, H, W)
        
        features = self.cnn(x)
        features = features.view(batch_size, num_frames, self.d_model)
        
        features = features + self.pos_embedding[:, :num_frames, :]
        
        transformer_out = self.transformer(features)
        
        global_pool = transformer_out.mean(dim=1)
        
        global_pool = self.dropout(global_pool)
        output = self.fc(global_pool)
        
        return output
        
        return x
