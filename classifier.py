import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import argparse


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


# Define the models as proper PyTorch classes
class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class SoftmaxModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.softmax(self.fc(x))

def train(dataset, n_hidden=50, batch_size=100, epochs=100, 
          learning_rate=0.01, model='nn', l2_ratio=1e-7, rtn_layer=True):
    train_x, train_y, test_x, test_y = dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert numpy arrays to PyTorch tensors
    train_x = torch.FloatTensor(train_x).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    n_in = train_x.shape[1]
    n_out = len(torch.unique(train_y))
    
    if batch_size > len(train_y):
        batch_size = len(train_y)
    
    print(f'Building model with {len(train_x)} training data, {n_out} classes...')
    
    # Initialize model
    if model == 'nn':
        print('Using neural network...')
        net = NeuralNetwork(n_in, n_hidden, n_out).to(device)
    else:
        print('Using softmax regression...')
        net = SoftmaxModel(n_in, n_out).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, 
                          weight_decay=l2_ratio)
    
    print('Training...')
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, train loss {round(total_loss, 3)}')
    
    # Evaluate training accuracy
    net.eval()
    with torch.no_grad():
        train_outputs = net(train_x)
        pred_y = torch.argmax(train_outputs, dim=1).cpu().numpy()
        true_y = train_y.cpu().numpy()
        
        print(f'Training Accuracy: {accuracy_score(true_y, pred_y)}')
        print(classification_report(true_y, pred_y))
        
        if test_x is not None:
            print('Testing...')
            test_x = torch.FloatTensor(test_x).to(device)
            test_y = torch.LongTensor(test_y).to(device)
            
            test_outputs = net(test_x)
            pred_y = torch.argmax(test_outputs, dim=1).cpu().numpy()
            true_y = test_y.cpu().numpy()
            
            print(f'Testing Accuracy: {accuracy_score(true_y, pred_y)}')
            print(classification_report(true_y, pred_y))
    
    if rtn_layer:
        return net
    else:
        return pred_y

def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(test_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(test_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(vars(args))
    dataset = load_dataset(args.train_feat, args.train_label, 
                          args.test_feat, args.train_label)
    train(dataset, model=args.model, learning_rate=args.learning_rate,
          batch_size=args.batch_size, n_hidden=args.n_hidden,
          epochs=args.epochs)

if __name__ == '__main__':
    main()
