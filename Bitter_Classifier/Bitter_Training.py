import os, shutil
from time import time
from BILN_Database import BILNDataset
from Bitter_AAF import BitterPepGCN, EarlyStopper
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt

N_EPOCHS = 2000
LR = 0.0001
REDUCE_LR_ON_PLATEAU = False
BATCH_SIZE = 128
NUM_WORKERS = 0
WD = 0.001
DROPOUT = 0.15
OUTPUTDIR = os.getcwd() + os.sep + 'Training_Results'

cwd = os.getcwd()

dataset = BILNDataset(root=os.path.join(cwd,'train'))

def get_rates(y_test,probs):
    fpr, tpr, thresholds = roc_curve(y_test, probs, drop_intermediate=False)
    auc_value = auc(fpr,tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return fpr,tpr,auc_value,optimal_threshold

# split the dataset into train and test
if os.path.exists(OUTPUTDIR):
    shutil.rmtree(OUTPUTDIR)
os.makedirs(OUTPUTDIR)

train_indexes = np.array(dataset.indices())
training_set = dataset[torch.tensor(train_indexes)]

aucs = []
train_aucs = []
start_time = time()
fpr_points = np.linspace(0,1,500)
tpr_train, tpr_val = [], []

#Define the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO   ] Device: {device}")

weights = None
for fold, (train_ids, val_ids) in enumerate(kfold.split(training_set,training_set.y)):
    image_name = OUTPUTDIR + os.sep + 'Images' + os.sep + f'fold_{fold+1}.png'
    
    fig,ax = plt.subplots(figsize=(8/2.54,8/2.54))
    
    if not os.path.exists(OUTPUTDIR + os.sep + 'Images'):
        os.makedirs(OUTPUTDIR + os.sep + 'Images')
    model_name = OUTPUTDIR + os.sep + f'fold_{fold+1}.pt'

    # Print split information:
    print(f'[INFO   ] Total Number of graphs: {len(train_ids)+len(val_ids)}')
    print(f"[INFO   ] Split information:") 
    print(f'[INFO   ] Number of training graphs: {len(train_ids)}')
    print(f'[INFO   ] Number of test graphs: {len(val_ids)}')

    # split the dataset according to the fold indices:
    train_ids = torch.tensor(train_ids)
    val_ids = torch.tensor(val_ids)
    train_dataset = training_set[train_ids]
    val_dataset = training_set[val_ids]   
    
    # Create the dataloaders:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=False)
    val_loader =  DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=False)

    #Create the model:
    model = BitterPepGCN(train_dataset.num_node_features,drop=DROPOUT)

    # Define loss function and optimizer
    model.criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # Define Reduce LR on Plateau:
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=15, min_lr=0.00005, verbose=True)

    # Define early stopper:
    early_stopper = EarlyStopper(patience=8, min_delta=0.03)

    # Move the model to the device:
    model = model.to(device)

    # Call forward with a dummy batch to initialize the parameters:
    dummy_batch = next(iter(train_loader))

    # Call forward with dummy batch:
    model(dummy_batch.to(device).x,dummy_batch.to(device).edge_index,batch=dummy_batch.to(device).batch)
    # Print model info:
    print("[INFO   ] Model information:")
    print('[INFO   ] ====================')
    print(f'[INFO   ] Number of parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'[INFO   ] Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # Define the training function:
    def train():
        model.train()
        loss_all = 0.0
        for data in train_loader:
            # Move to the device
            data = data.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Predict output and calculate loss:
            output = model(data.x, data.edge_index, batch=data.batch)
            if data.num_graphs > 1:
                loss = model.criterion(output, data.y.type(torch.LongTensor).to(device))
            else:
                loss = model.criterion(output.unsqueeze(0), data.y.type(torch.LongTensor).to(device))
            # Backpropagation
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            # Step:
            optimizer.step()
            del data
        return loss_all / len(train_dataset)

    # Define the test function:
    def test(loader):   
        model.eval()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, batch=data.batch)
            if data.num_graphs > 1:
                loss = model.criterion(output, data.y.type(torch.LongTensor).to(device))
            else:
                loss = model.criterion(output.unsqueeze(0), data.y.type(torch.LongTensor).to(device))
            loss_all += data.num_graphs * loss.item()
            del data
        return loss_all / len(loader.dataset)

    # Train the model:
    print("[INFO   ] Training the model...")

    # Initialize arrays to store train loss and test loss:
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, N_EPOCHS+1):
        train_loss = train()
        test_loss = test(val_loader)
        last_epoch = epoch
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
        # Append to array for plotting:
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)

        # Early stopping if val_loss diverges
        if early_stopper.early_stop(test_loss,epoch=epoch):
            print(f'Early stopping at Epoch {epoch:03d} to avoid overfitting, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
            break

        # Reduce lr on plateau
        if REDUCE_LR_ON_PLATEAU:
            scheduler.step(test_loss)
            
    # Plot loss:
    fig,ax = plt.subplots(figsize=(8/2.54,8/2.54))
    ax.plot(range(1,last_epoch+1),train_loss_list,label='Train',color='navy')
    ax.plot(range(1,last_epoch+1),val_loss_list,label='Test',color='darkred')
    ax.legend(frameon=False)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # save the figure
    fig.tight_layout()
    fig.savefig(image_name,dpi=600)
    plt.close(fig)

    # Save the model:
    torch.save(model.state_dict(), model_name)

    # Evaluate the model with the roc curve:
    model.eval()
    y_train, train_probs = [], []
    for data in train_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        try:
            train_probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        except IndexError:
            output = output.unsqueeze(0)
            train_probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        y_train.append(data.y.cpu().detach().numpy())
        del data
    train_probs = np.concatenate(train_probs)
    y_train = np.concatenate(y_train)
    t_fpr,t_tpr,t_auc_value,_ = get_rates(y_train,train_probs)
    # Plot the roc curve for training:
    if t_auc_value > 0.55:
        tpr_points = np.interp(fpr_points, t_fpr, t_tpr)
        tpr_train.append(tpr_points)

    train_aucs.append(t_auc_value)
    del y_train, train_probs

    y_val, probs = [], []
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        probs.append(F.softmax(output,dim=1)[:,1].cpu().detach().numpy())
        y_val.append(data.y.cpu().detach().numpy())
        del data
    probs = np.concatenate(probs)
    y_val = np.concatenate(y_val)
    fpr,tpr,auc_value,_ = get_rates(y_val,probs)
    if auc_value > 0.55:
        # ax_roc = plot_roc(fpr,tpr,ax_roc,color='navy',alpha=0.5)
        tpr_points = np.interp(fpr_points, fpr, tpr)
        tpr_val.append(tpr_points)
    aucs.append(auc_value)
    del y_val, probs
    print(f'[INFO   ] Model {fold+1} saved.')

aucs = np.array(aucs)

print(f'[INFO   ] Mean AUC on Training: {np.mean(train_aucs_clean):.2f} +/- {np.std(train_aucs_clean):.2f}')
print(f'[INFO   ] Mean AUC on Test: {np.mean(aucs_clean):.2f} +/- {np.std(aucs_clean):.2f}')

end_time = time()
print('[INFO   ] Finished Training in {} minutes, num_workers={}'.format(round((end_time - start_time)/60,2), NUM_WORKERS) )

# get the best model and load it:
best_model_ids = np.argmax(aucs)
best_model_name = OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}.pt'
model.load_state_dict(torch.load(best_model_name))
model.to(device)
os.rename(OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}.pt', OUTPUTDIR + os.sep + f'fold_{best_model_ids+1}_BEST.pt')
print(f'[INFO   ] Best Model is at fold {best_model_ids+1}')

# Plot the roc curve:
train_mean_tpr, train_std_tpr = np.vstack(tpr_train).mean(axis=0), np.vstack(tpr_train).std(axis=0)
val_mean_tpr, val_std_tpr = np.vstack(tpr_val).mean(axis=0), np.vstack(tpr_val).std(axis=0)
fig_roc,ax_roc = plt.subplots(figsize=(8/2.54,8/2.54))
ax_roc.plot(fpr_points, train_mean_tpr, color='steelblue',
            label=r'Train (AUC = {} $\pm$ {})'.format(np.round(np.mean(train_aucs_clean),2),np.round(np.std(train_aucs_clean),2)),alpha=1)
ax_roc.fill_between(fpr_points, train_mean_tpr - train_std_tpr, train_mean_tpr + train_std_tpr, facecolor='steelblue', edgecolor=None, alpha=0.2)
ax_roc.plot(fpr_points, val_mean_tpr, color='darkgreen',
            label=r'Test (AUC = {} $\pm$ {})'.format(np.round(np.mean(aucs_clean),2),np.round(np.std(aucs_clean),2)),alpha=1)
ax_roc.fill_between(fpr_points, val_mean_tpr - val_std_tpr, val_mean_tpr + val_std_tpr, facecolor='darkgreen', edgecolor=None, alpha=0.2)
ax_roc.plot([0, 1], [0, 1],'k--',zorder=0)
ax_roc.set_xlim([-0.05, 1.05])
ax_roc.set_ylim([-0.05, 1.05])
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
ax_roc.legend(frameon=False,loc=(0.3,0),fontsize=8)
fig_roc.tight_layout()
roc_name = OUTPUTDIR + os.sep + 'Images' + os.sep + f'roc_curve.png'
fig_roc.savefig(roc_name,dpi=600,facecolor='white')
roc_name_svg = OUTPUTDIR + os.sep + 'Images' + os.sep + f'roc_curve.svg'
fig_roc.savefig(roc_name_svg,dpi=600,facecolor='white')
plt.close(fig_roc)

print('[DONE   ] Model trained. All folds were saved to "Models" folder')