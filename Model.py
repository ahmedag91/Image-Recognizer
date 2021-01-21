import torch
from torch import nn, optim
from workspace_utils import active_session
import time

class Network(nn.Module):
    

    def __init__(self, input_dims = 25088, hidden_layers = [4096], output_dims = 102, dropout = 0.2):
        super().__init__()
        

        # define an empy_list
        l = [input_dims]
        for dim in hidden_layers:
            l.append(dim)
        l.append(output_dims)


        # add the hidden and layers
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.extend([nn.Linear(l[i], l[i+1]) for i in range(len(l)-1)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        print(len(self.hidden_layers))
    def forward(self, network_input):

        for layer in self.hidden_layers[:-1]:
            network_input = layer(network_input)
            network_input = self.activation(network_input)
            network_input = self.dropout(network_input)
        network_input = hidden_layers(network_input)
        return nn.LogSoftmax(network_input)
def Extend(model1, model2):
    
    for layer in model1.parameters():
        layer.requires_grad = False
    model1.classifier = model2

    return model1

def train_model(model, train_data, valid_data, epochs = 10, lr = 0.005, device = torch.device('cpu')):
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()

    with active_session():
        for e in range(epochs):
            
            # restart the timer
            start = time.time()
            
            # Make the model in the training mode to consider the dropouts
            model.train()

            # Set the training_loss to zero at each epoch
            training_loss = 0
            for images, labels in train_data:
                
                # Move the images and the labels to the device
                images = images.to(device)
                labels = labels.to(device)

                # reset the gradient to zero for prevention from accumulation
                optimizer.zero_grad()

                # Estimate the logarithm of the propabilities (Check the output layer of the classifer)
                log_props = model.forward(images)

                # Claculate the loss function
                loss = criterion(log_props, labels)

                # Update the training_loss
                training_loss += loss.item()

                #Backpropagation
                loss.backward()

                # Update the weights
                optimizer.step()

            # The validation phase

            with torch.no_grad():
                # enter the evaluation mode to deactivate the dropouts
                model.eval()
                
                # Set the validation loss and accuracy to zero
                validation_loss = 0
                validation_accuracy = 0
                

                for images, labels in valid_data:
                    # Move the images and the labels to the device
                    images = images.to(device)
                    labels = labels.to(device)

                    # Estimate the logarithm of the propabilities (Check the output layer of the classifer)
                    log_props = model.forward(images)

                    # Update the validation loss
                    validation_loss += criterion(log_props, labels).item()

                    # Compute the propabilities
                    props = torch.exp(log_props, dim = 1)

                    # Compute the top classes corresponding to the top propabilities
                    _, top_classes = props.topk(1, dim = 1)

                    
                    # Compute which labels wer predicted correctly in the batch
                    equals = top_classes == labels.view(*top_classes.shape)

                    # take the average accuray per batch and update the accumulated validation_accuracy
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            end = time.time()

            # Print the parameters to keep yourself updated
            print("Epoch {}/{} ===".format(e+1, epochs), 
              "Training loss: {:.3f} ===".format(training_loss/len(train_data)), 
              "Validation loss: {:.3f} ===".format(validation_loss/len(valid_data)), 
              "Validation accuracy: {:.3f}% ===".format(validation_accuracy*100/len(valid_data)),
              'Time consumed: {:.0f} seconds'.format(end-start))






    return model
