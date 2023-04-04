import json
import pandas as pd

path_inference_result = r'Z:\bak_sgb\custom_prediction.json'
with open(path_inference_result, 'r') as f:
    image_data = json.load(f)
    image_data = pd.DataFrame(image_data)
    test = 1

# len(image_data.iloc[-1,0][0])