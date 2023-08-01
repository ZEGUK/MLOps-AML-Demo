import os
from azureml.core.model import Model
import argparse
from azureml.core import Workspace

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, dest = 'evaluated_data', default = 'evaluated_data', help='evaluated_data')
args = parser.parse_args()
evaluated_data = args.evaluated_data

# get the model path
model_path = os.path.join(evaluated_data, "model_for_predicting.pth")
# Optional for better tracking model files in the working dir
# model_path = os.path.join('datafiles/predicting_data', 'model_for_predicting.pth')

# Register the model
ws = Workspace.from_config()
model = Model.register(ws,model_name = '<YOUR_MODEL_NAME>', model_path = model_path)
print(model.name, model.id, model.version, sep='\t')
