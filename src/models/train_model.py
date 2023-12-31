from torch.utils.data import Dataset
import torch
import os
import mlflow.pytorch
from mlflow import MlflowClient
from dotenv import load_dotenv
import torch.nn as nn
from codecarbon import EmissionsTracker
from torchvision import transforms
from model import SimpleCNNReducedStride10
import sys

class CustomDataset(Dataset):
    """
    This is a class that extends the Dataset class from pytorch. It's used to make sure that the 
    input for the model is properly initialized. This means resizing it to be 500 x 500, initialising it
    as a tensor and normalizing the RGB values. 
    """
    def __init__(self, dataset):
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.Resize((500,500)),  # Resize to our desired size
            transforms.ToTensor(),          # Convert PIL Image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB channels
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform(sample['image'])
        label = sample['labels']

        return image, label
    
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def main():

    # Get the absolute path to the root of your project
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(os.path.join(project_root,'.env'))
    load_dotenv(dotenv_path=os.path.join(project_root,'.env'))

    train_loader = torch.load(os.path.join(project_root, 'data', 'processed', 'train_loader.pt'))
    validation_loader= torch.load(os.path.join(project_root,'data', 'processed', 'validation_loader.pt'))

    # Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
    mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME')
    if mlflow_tracking_username is not None:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
    else:
        print("Warning: MLFLOW_TRACKING_USERNAME is not defined")

    mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if mlflow_tracking_password is not None:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password
    else:
        print("Warning: MLFLOW_TRACKING_PASSWORD is not defined")

    mlflow.set_experiment("CNN-pytorch")
    mlflow.pytorch.autolog
    mlflow.set_tracking_uri('https://dagshub.com/wwoszczek/MLOps-TeamBeans.mlflow')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(sys.argv) > 1:
        emission_choice = sys.argv[1]
    else:
        emission_choice = input(
            'Do you want to track emissions from the model training process? Press 1 for yes and 2 for no:  ')
    if int(emission_choice) == 1:
        tracker = EmissionsTracker(output_dir=os.path.join(project_root, 'models'))
        tracker.start()
        
    with mlflow.start_run() as run:
                
        # Create an instance of the SimpleCNNReduced model
        model = SimpleCNNReducedStride10(num_classes=3)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate the total number of trainable parameters
        total_params_reduced = count_parameters(model)
        mlflow.log_param("total_trainable_parameters", total_params_reduced)
        print(f"Total trainable parameters in the reduced model: {total_params_reduced}")
        
        ###############################3
        
        from torch.optim import Adam

        model = model.to(device)
        optimizer = Adam(model.parameters())
        criterion = nn.NLLLoss()

        num_epochs = 5
        batch_loss = 0
        cum_epoch_loss = 0
        
        # Log parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("num_classes", 3)
        mlflow.log_param("kernel_size_conv1", 3)
        mlflow.log_param("stride_conv1", 2)
        mlflow.log_param("padding_conv1", 1)
        mlflow.log_param("kernel_size_conv2", 3)
        mlflow.log_param("stride_conv2", 2)
        mlflow.log_param("padding_conv2", 1)
        mlflow.log_param("dropout_rate", 0.5)
        mlflow.log_param("fc1_input_size", model.fc1_input_size)
        mlflow.log_param("num_conv_layers", 2)  # Example: Number of convolutional layers
        mlflow.log_param("activation_function", "ReLU")  # Example: Activation function used
        for e in range(num_epochs):
            cum_epoch_loss = 0
            list_epoch_loss=[]
            individual_loss=0
            for batch, (images, labels) in enumerate(train_loader,1):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                individual_loss += loss.item()
                batch_loss += loss.item()
                print(f'Epoch({e+1}/{num_epochs} : Batch number({batch}/{len(train_loader)}) : Batch loss : {loss.item()}')

            epoch_loss = batch_loss / len(train_loader)
            list_epoch_loss.append(individual_loss/ len(train_loader))
            cum_epoch_loss += epoch_loss

            # Log the cumm and individual epoch loss as a metric
            mlflow.log_metric("cummulative_epoch_loss", epoch_loss)
            # mlflow.log_metric("individual_epoch_loss", individual_loss/ len(train_loader))
            mlflow.log_metric("individual_epoch_loss_"+str(e+1),individual_loss/ len(train_loader))
            print(f'Epoch({e + 1}/{num_epochs}) : Cummulative Epoch loss: {epoch_loss}')
            print(f'Epoch({e + 1}/{num_epochs}) : Individual Epoch loss: {individual_loss/ len(train_loader)}')

        print(f'Training loss : {batch_loss/len(train_loader)}')
        
        # Log a metric (e.g., training loss)
        mlflow.log_metric("training_loss", batch_loss / len(train_loader))
        
        model.to('cpu')
        
        # Save the model as an artifact
        mlflow.pytorch.log_model(model, "models")

        model.eval()
        with torch.no_grad():
            num_correct = 0
            total = 0

            #set_trace()
            for batch, (images, labels) in enumerate(validation_loader,1):

                logps = model(images)
                output = torch.exp(logps)

                pred = torch.argmax(output, 1)
                total += labels.size(0)
                num_correct += (pred == labels).sum().item()
                print(f'Batch ({batch}/{len(validation_loader)})')

                # if batch == 5:
                # break

            # Calculate test accuracy
            val_accuracy = num_correct * 100 / total
            print(f'Accuracy of the model on {total} validation images: {val_accuracy}% ')

            # Log the test accuracy as a metric
            mlflow.log_metric("validation_accuracy", val_accuracy)
                    
    
    # save the trained model
    torch.save(model.state_dict(), os.path.join(project_root, 'models', 'trained_model.pt'))
            
    # fetch the auto logged parameters and metrics
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        
    if int(emission_choice) == 1:
        tracker.stop()
    
if __name__=='__main__':
    main()  