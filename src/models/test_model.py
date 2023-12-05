import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.model import SimpleCNNReducedStride10

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
import csv

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


def main():
    
    # Get the absolute path to the root of your project
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_loader = torch.load(os.path.join(project_root,'data', 'processed', 'test_loader.pt'))


    # Initialize the pytorch model
    model=SimpleCNNReducedStride10()

    path=os.path.join(project_root,'models')
    for filename in os.scandir(path):
        if "model.pt" in str(filename):
            model.load_state_dict(torch.load(os.path.join(path,filename)))
            model.eval()

    with torch.no_grad():
            total = 0
            all_preds = []
            all_labels = []

            for batch, (images, labels) in enumerate(test_loader,1):

                logps = model(images)
                output = torch.exp(logps)

                pred = torch.argmax(output, 1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)

            # Calculate test accuracy
            test_accuracy = accuracy_score(all_labels, all_preds)
            print(f'Accuracy of the model on {total} test images: {test_accuracy * 100:.2f}%')

            # Calculate precision, recall, and F1 score per class
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

            for i in range(len(precision)):
                print(f'Class {i + 1}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1 Score={f1[i]:.4f}')

            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            print('Confusion Matrix:')
            print(cm)

            # Store metrics in a CSV file
            csv_file_path = os.path.join(project_root, 'models', 'test_metrics.csv')
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write header
                csv_writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'])

                # Write metrics for each class
                for i in range(len(precision)):
                    # Convert confusion matrix row to a string
                    confusion_matrix_str = ','.join(map(str, cm[i]))
                    csv_writer.writerow([f'Class {i + 1}', precision[i], recall[i], f1[i], confusion_matrix_str])

            print(f'Metrics saved to: {csv_file_path}')


if __name__=='__main__':
    main()  