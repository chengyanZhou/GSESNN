import torch
import numpy as np
from sklearn import metrics
from torch.optim import Adam
from torch.optim import lr_scheduler
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

def train_epochs(train_dataset, test_dataset, feat_di_sim, feat_dr_sim, drug_adj, disease_adj, model, args):

    num_workers = 2
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=num_workers)

    test_size = 1024
    test_loader = DataLoader(test_dataset, test_size, shuffle=False,
                             num_workers=num_workers)

    model.to(args.device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)
    best_epoch, best_auc, best_aupr = 0, 0, 0
    for epoch in pbar:
        train_loss = train(model, feat_di_sim, feat_dr_sim, drug_adj, disease_adj, optimizer, train_loader, args.device, args.p)
        if epoch % args.valid_interval == 0:
            roc_auc, aupr = evaluate_metric(model, feat_di_sim, feat_dr_sim, drug_adj, disease_adj, test_loader, args.device, args.p)
            print("epoch {}".format(epoch), "train_loss {0:.4f}".format(train_loss),
                  "roc_auc {0:.4f}".format(roc_auc), "aupr {0:.4f}".format(aupr))
            if roc_auc > best_auc:
                best_epoch, best_auc, best_aupr = epoch, roc_auc, aupr

    print("best_epoch {}".format(best_epoch), "best_auc {0:.4f}".format(best_auc), "aupr {0:.4f}".format(best_aupr))
   
    return best_auc, best_aupr

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, feat_di_sim, feat_dr_sim, drug_adj, disease_adj, optimizer, loader, device, p):
    model.train()
    total_loss = 0
    pbar = loader
    for data in pbar:      
        optimizer.zero_grad()
        true_label = data.to(device)
        predict = model(true_label, torch.FloatTensor(feat_di_sim).to(device), 
                        torch.FloatTensor(feat_dr_sim).to(device),
                        torch.FloatTensor(drug_adj).to(device), 
                        torch.FloatTensor(disease_adj).to(device), p)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(predict, true_label.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
     
    return total_loss / len(loader.dataset)


def evaluate_metric(model, feat_di_sim, feat_dr_sim, drug_adj, disease_adj, loader, device, p):
    model.eval()
    pbar = loader
    roc_auc, aupr = None, None
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data, torch.FloatTensor(feat_di_sim).to(device), 
                        torch.FloatTensor(feat_dr_sim).to(device), 
                        torch.FloatTensor(drug_adj).to(device), 
                        torch.FloatTensor(disease_adj).to(device), p)

        y_true = data.y.view(-1).cpu().tolist()
        y_score = out.cpu().numpy().tolist()

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)

        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(recall, precision)
        torch.cuda.empty_cache()
  

    return roc_auc, aupr
