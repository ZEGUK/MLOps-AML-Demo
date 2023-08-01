import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import argparse
import os

# 11/25 coding by ZEGUK
#######################################################################################################################################
# # Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_for_evaluation", type = str, dest = 'trained_data', default = 'trained_data', help='trained data')
parser.add_argument("--model_for_predicting", type = str, dest = 'evaluated_data', default = 'evaluated_data', help='evaluated data')
args = parser.parse_args()
trained_data = args.trained_data
evaluated_data = args.evaluated_data
""
datafiles_dir = "<YOUR_FILES_DIR>"
EVAL_DIR = os.path.join(datafiles_dir,'datafiles/evaluating_data/')# Note that Pytorch generally reads files on the father dir!
EVAL_MODEL = os.path.join(trained_data, "model.pth")
# EVAL_DIR= "datasets/eval/fail/"
# EVAL_MODEL= "RC_trial_2.pth"

model = torch.load(EVAL_MODEL)
model.eval()

num_cpu = multiprocessing.cpu_count()    # 0
bs = 16

eval_transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(eval_dataset.classes)
dsize = len(eval_dataset)

class_names = ['fail', 'pass']

predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

overall_accuracy = 100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))

conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix\n')
print(conf_mat,'\n')

class_accuracy = 100 * conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy\n')
for label,accuracy in zip(eval_dataset.classes, class_accuracy):
    # Modified on 12.04 by ZEGUK: int(label)is ERRO since label = 'fail' not a str(0) for example.
    #class_name=class_names[int(label)]
    #print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))
    print('Accuracy of class %8s : %0.2f %%'%(label, accuracy))
##################################################################################################################################
# Modified on 12.04 by MSFT-CSU-CE-LU HUO
if overall_accuracy > 0.7:
    print("\nSaving the model for predicting")
    os.makedirs(evaluated_data, exist_ok=True)
    model_for_predicting_path = os.path.join(evaluated_data,'model_for_predicting.pth')
    torch.save(model, model_for_predicting_path)
    # Optional for better tracking model files in the working dir
    # os.makedirs('datafiles/predicting_data', exist_ok = True)
    # model_path = os.path.join('datafiles/predicting_data/', 'model_for_predicting.pth')
    # torch.save(model, model_path)
    print("\nSaving Completed!")
