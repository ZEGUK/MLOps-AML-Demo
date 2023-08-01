from azureml.monitoring import ModelDataCollector
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.core import Workspace
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import pandas as pd


######################################################################################################################
# Link to the worksapce when called
svc_pr_password = os.getenv("AZUREML_PASSWORD")# Keep your service principal password invisuable. Set the env variable during deployment 
svc_pr = ServicePrincipalAuthentication( tenant_id="YOUR_TENANT_ID", 
                                        service_principal_id="YOUR_SERVICE_PRINCIPAL_ID", 
                                           service_principal_password=svc_pr_password)

ws = Workspace( subscription_id="YOUR_SUBSCRIPTION_ID", 
               resource_group="YOUR_RESOURCE_GROUP", 
               workspace_name="YOUR_WORKSPACE_NAME", 
               auth=svc_pr )
#######################################################################################################################
# init() and run() are vital parts in the scoring script. init() generally for load model and run() for inference.

def init():
    global model
    
    print("This is init()")
    model_path = Model.get_model_path(model_name = 'YOUR_MODEL_NAME', version = 1, _workspace = ws, _location = 'YOUR_LOCATION')
    # Or another way to get model path
    # model_filename = 'model_for_predicting.pth'
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_filename)
    model = torch.load(model_path)

    global inputs_dc, prediction_dc
    inputs_dc = ModelDataCollector("best_model", designation="inputs", 
                                    feature_names=["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"])
    prediction_dc = ModelDataCollector("best_model", designation="predictions", 
                                        feature_names=["prediction1", "prediction2"])
    
    
def preprocessing(image):
    preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = preprocess(image).unsqueeze(0).to(device)
    return inputs


@rawhttp
def run(request):
    
    print("This is run()")
    
    if request.method == 'GET':
        # For this example, just return the URL for GETs.
        respBody = str.encode(request.full_path)
        return AMLResponse(respBody, 200)
    elif request.method == 'POST':
        print(AMLResponse(request.method, 200))
        file_bytes = request.files["image"]
        image = Image.open(file_bytes).convert('RGB')
        inputs = preprocessing(image)
        class_names=['fail', 'pass']
        pred_result = pd.DataFrame(columns = ['pred_class'])
        model.eval()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        label=class_names[preds]
        pred_result = pred_result.append([{'pred_class':label}], ignore_index=True)
        print("pred result: ", pred_result.to_json())
        # inputs_dc.collect(inputs) #this call is saving our input data into Azure Blob
        # prediction_dc.collect(pred_result) #this call is saving our prediction data into Azure Blob
        
        # return AMLResponse(json.dumps(pred_result), 200)
        return AMLResponse(pred_result.to_json(), 200)
        
    else:
        # return AMLResponse("bad request", 500)
        return AMLResponse(request.method, 500)


