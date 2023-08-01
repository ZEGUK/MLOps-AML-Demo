import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
import argparse
from azureml.core import Workspace, Datastore, Dataset

# ap = argparse.ArgumentParser()
# ap.add_argument("--mode", required=True, help="Training mode: finetue/transfer/scratch")
# args= vars(ap.parse_args())

# 11/25 coding by ZEGUK
""
ws = Workspace.from_config()
print("Workspace and location: " + ws.name + ws.location)
cwd = os.getcwd()
print("Working Dir" + cwd)

# Get datastore and download files to local Notebooks file folders(including training data and evaluating data)
datastore = Datastore.get(ws, '<YOUR_DATASTORE_NAME>')

datafiles_dir = '<YOUR_FILES_DIR>'

training_data_datastore_paths = [(datastore, 'training_dataset')]
training_data_ds = Dataset.File.from_files(path = training_data_datastore_paths)
os.makedirs(os.path.join(datafiles_dir, 'datafiles/training_data'), exist_ok = True)
training_data_ds.download(target_path = os.path.join(datafiles_dir, 'datafiles/training_data'), overwrite=True)

evaluating_data_datastore_paths = [(datastore, 'evaluating_dataset')]
evaluating_data_ds = Dataset.File.from_files(path = evaluating_data_datastore_paths)
os.makedirs(os.path.join(datafiles_dir, 'datafiles/evaluating_data'), exist_ok = True)
evaluating_data_ds.download(target_path = os.path.join(datafiles_dir, 'datafiles/evaluating_data'), overwrite=True)

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--output_model_dir", type=str, dest='trained_data', default = 'trained_data', help='output_model')
args = parser.parse_args()
trained_data = args.trained_data
os.makedirs(trained_data, exist_ok=True)
# os.makedirs('datafiles/trained_data/', exist_ok = True) # for debugging or testing script

""
train_directory = os.path.join(datafiles_dir, 'datafiles/training_data/train/')
valid_directory = os.path.join(datafiles_dir, 'datafiles/training_data/val/')
PATH = os.path.join(trained_data, "model.pth")

train_mode="0"

# train_directory = "datasets/train/"
# valid_directory = "datasets/val/"
# PATH= "model.pth"

bs = 32
num_epochs = 10
num_classes = 2
num_cpu = multiprocessing.cpu_count()    # 0

image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True)
}

class_names = dataset['train'].classes
print("Classes:", class_names)

print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_mode=='0':
    print("\nLoading resnet\n")
    model_ft = models.resnet18(pretrained=False)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes )

elif train_mode=='1':
    print("\nLoading mobilenet\n")
    model_ft = models.mobilenet_v2(pretrained=False)

    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        )


model_ft = model_ft.to(device)

print('Model Summary:\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )
summary(model_ft, input_size=(3, 224, 224))
print(model_ft)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("\nTraining:\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)

print("\nSaving the model")
#Optional for better tracking model files in the working dir
#torch.save(model_ft, 'datafiles/trained_data/model.pth')
torch.save(model_ft, PATH)
