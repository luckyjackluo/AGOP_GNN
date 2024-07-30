import torch
import torch.nn.functional as F

def train(data, task, model, optimizer, criterion, device):
    assert task in ["node_classify", "node_regress", "graph_regress"]
    model.train()
    if task in ["node_classify", "node_regress"]:
        data, train_idx = data
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx].squeeze(1))
        loss.backward()
        optimizer.step()
        return loss.item()
    
    elif task in ["graph_regress"]:
        total_loss = 0
        train_loader = data
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, batch=data.batch)
            loss = criterion(out, data.y[:, 10].reshape(out.shape))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader.dataset)

def test(data, task, model, criterion, device):
    assert task in ["node_classify", "node_regress", "graph_regress"]
    model.eval()
    
    if task in ["node_classify"]:
        data, train_idx, val_idx, test_idx = data
        out = model(data.x, data.edge_index)
        y_pred = out.argmax(dim=-1, keepdim=True)
        train_acc = (y_pred[train_idx] == data.y[train_idx]).sum().item() / train_idx.size(0)
        val_acc = (y_pred[val_idx] == data.y[val_idx]).sum().item() / val_idx.size(0)
        test_acc = (y_pred[test_idx] == data.y[test_idx]).sum().item() / test_idx.size(0)
        return train_acc, val_acc, test_acc
    
    else:
        val_loader, test_loader = data
        total_val_loss = 0
        total_test_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, batch=data.batch)
                total_val_loss += F.l1_loss(out, data.y[:, 10].reshape(out.shape)).item()
                
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, batch=data.batch)
                total_test_loss += F.l1_loss(out, data.y[:, 10].reshape(out.shape)).item()
                
        return 0, total_val_loss/len(val_loader), total_test_loss/len(test_loader)
