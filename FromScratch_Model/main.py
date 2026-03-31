#---------------------------------------------< Imports >------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_load import create_dataloaders
from src.model import ThreeDCNN, CNNRNN, VideoTransformer
from src.train import train_model
import argparse








#------------------------------------------------------------------------< Fucntions >------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Shoplifting Detection Training')
    parser.add_argument('--model', type=str, default='3dcnn', choices=['3dcnn', 'cnnrnn', 'transformer'],
                        help='Model architecture to use')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing shop lifters and non shop lifters folders')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        test_split=0.2,
        num_workers=2
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    if args.model == '3dcnn':
        model = ThreeDCNN(num_classes=2)
        print('Using 3D CNN model')
    elif args.model == 'cnnrnn':
        model = CNNRNN(num_classes=2, hidden_size=128, num_layers=2)
        print('Using CNN + RNN model')
    elif args.model == 'transformer':
        model = VideoTransformer(num_classes=2, d_model=512, nhead=8, num_layers=4)
        print('Using Transformer model')
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    results = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=os.path.join(args.save_dir, args.model)
    )
    
    print(f'\nTraining completed! Best test accuracy: {results["best_test_acc"]:.4f}')

#------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()


