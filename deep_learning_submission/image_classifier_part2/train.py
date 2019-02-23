import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import utils
import constants

def train(args):
    """
        train model based on given args
    """
    # Get args
    data_dir = args.data_dir
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    epochs = args.epochs
    gpu = args.gpu
    arch = args.arch
    save_dir = args.save_dir
    
    device = torch.device("cuda" if gpu else "cpu")
    
    model = utils.vgg_model_select(name=arch)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # replace pretrained model classifier with our classifier. 
    model.classifier = classifier

    # define an optimizer.
    # Adam is a fair choice and in general quicker to train
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # define criterion NLLLoss
    # NegativeLikelihoodLogLoss + LogSoftmax ~ CrossEntropyLoss
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.functional.binary_cross_entropy
    criterion = nn.NLLLoss()
    
    # training and validation datas and data loaders
    train_data, train_loader = utils.data_loader(data_dir, dataset='train', batch_size=64)
    valid_data, valid_loader = utils.data_loader(data_dir, dataset='validation', batch_size=64)
    
    # Start training the model    
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = constants.PRINT_EVERY
    
    print("Training {arch} model for {epochs} epoch(s).\n "
          "# of hidden perceptrons: {hidden_units} \n"
          "Learning Rate: {learning_rate}\n"
          "Training with GPU set to {gpu}".format(arch=arch,
                                                  epochs=epochs,
                                                  hidden_units=hidden_units,
                                                  learning_rate=learning_rate,
                                                  gpu=gpu))
          
    for epoch in range(epochs):
        for inputs, labels in train_loader:
        
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            steps += 1
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    
    print('Finished training model!, saving model to dir: {0}'.format(save_dir))
    name = 'trained_model_{0}.pth'.format(arch)
    utils.save_model(model, epoch, optimizer, train_data, save_dir, name=name)
    print('Model saved to {0}/{1}'.format(save_dir, name))
    
if __name__ == '__main__':
    args = utils.train_argparser()
    train(args)
    