#-----------------------------------------< Imports >------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix







#-------------------------------------------------------------------< Functions >--------------------------------------------------------------------------


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

#----------------------------------------------------------------------------
def test_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary')
    epoch_recall = recall_score(all_labels, all_preds, average='binary')
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, all_preds, all_labels

#------------------------------------------------------------------------------------------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=50, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter('runs/experiment')
    
    best_test_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        print(f'Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        
        if (epoch + 1) % 5 == 0:
            test_loss, test_acc, test_precision, test_recall, test_f1, _, _ = test_epoch(model, test_loader, criterion, device)
            
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Precision/test', test_precision, epoch)
            writer.add_scalar('Recall/test', test_recall, epoch)
            writer.add_scalar('F1/test', test_f1, epoch)
            
            print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
            print(f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}')
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_f1': test_f1,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'Best model saved! (Test Acc: {test_acc:.4f})')
    
    writer.close()
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc
    }
