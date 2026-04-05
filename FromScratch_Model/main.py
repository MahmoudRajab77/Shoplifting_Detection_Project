#---------------------------------------------< Imports >------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_load import create_dataloaders
from src.model import ThreeDCNN, CNNRNN, VideoTransformer
from src.train import train_model








#------------------------------------------------------------------------< Fucntions >------------------------------------------------------------------------

def main():
    DATA_ROOT = '/kaggle/input/datasets/mahmoudrajab/shoplifting-dataset/Shop DataSet/Shop DataSet'
    BATCH_SIZE = 16
    NUM_FRAMES = 20
    EPOCHS = 20
    LR = 0.0001
    SAVE_DIR = 'checkpoints'
    MODEL_TYPE = 'cnnrnn'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_frames=NUM_FRAMES,
        test_split=0.2,
        num_workers=2
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    if MODEL_TYPE == '3dcnn':
        model = ThreeDCNN(num_classes=2)
        print('Using 3D CNN model')
    elif MODEL_TYPE == 'cnnrnn':
        model = CNNRNN(num_classes=2, hidden_size=128, num_layers=2)
        print('Using CNN + RNN model')
    elif MODEL_TYPE == 'transformer':
        model = VideoTransformer(num_classes=2, d_model=512, nhead=8, num_layers=4)
        print('Using Transformer model')
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=EPOCHS,
        save_dir=os.path.join(SAVE_DIR, MODEL_TYPE),
        patience=10
    )
    
    print(f'\nTraining completed! Best test accuracy: {results["best_test_acc"]:.4f}')

#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()


