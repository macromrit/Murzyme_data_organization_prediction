from fastapi import FastAPI
import uvicorn
import json
import numpy as np
import pickle as pkl


app = FastAPI()

# Output Mapping
OUTPUT_MAPPING = {0: "Non-Murzyme", 1: "Murzyme"}

# Loading the ML model
with open('murzyme_classification_svm.pkl', 'rb') as modelHandler:
    model = pkl.load(modelHandler)

# Feature Scaling model
with open("scaler_model.pkl", "rb") as modelScaler:
    scaler = pkl.load(modelScaler)

# Feature Extraction and Engineering
def extract_features(arr):
    res = list()
    for vctr in arr:
        if type(arr) is not type(np.array([])) or type(arr) is not list:
            res += ([0] * 7) # += is as same as .extend() method
            continue
        # else
        res += [
            np.mean(arr),
            np.std(arr),
            np.min(arr),
            np.max(arr),
            np.median(arr),
            np.percentile(arr, 25),  # First quartile (Q1)
            np.percentile(arr, 75)   # Third quartile (Q3)
        ]
    return res

def extractAndScaleAndPredict(input):
    extracted_features = extract_features(input)
    scaled_features = scaler.transform([extracted_features])
    print(scaled_features)
    return OUTPUT_MAPPING[list(model.predict(scaled_features))[0]]


# Loading data from DB(JSON)
with open("./raw_data/murzymes.pkl", "rb") as jammer, open("./raw_data/nonmurzymes.pkl", "rb") as hammer:
    rawMurzymes = pkl.load(jammer)
    rawNonMurzymes = pkl.load(hammer)


# Reading TNSE Plot readings from DB
with (open("tnse/murzymes/feature1_murzymes_tnse.json", 'r') as tnse_mf1, # tnse plot - murzymes - feature - 1
      open("tnse/murzymes/feature2_murzymes_tnse.json", 'r') as tnse_mf2, 
      open("tnse/murzymes/feature3_murzymes_tnse.json", 'r') as tnse_mf3, 
      open("tnse/murzymes/feature4_murzymes_tnse.json", 'r') as tnse_mf4, 
      open("tnse/non_murzymes/feature1_non_murzymes_tnse.json", 'r') as tnse_nmf1, 
      open("tnse/non_murzymes/feature2_non_murzymes_tnse.json", 'r') as tnse_nmf2, 
      open("tnse/non_murzymes/feature3_non_murzymes_tnse.json", 'r') as tnse_nmf3, 
      open("tnse/non_murzymes/feature4_non_murzymes_tnse.json", 'r') as tnse_nmf4, 
      ):

    tnse_mf1_xy = json.loads(tnse_mf1.read())
    tnse_mf2_xy = json.loads(tnse_mf2.read())
    tnse_mf3_xy = json.loads(tnse_mf3.read())
    tnse_mf4_xy = json.loads(tnse_mf4.read())

    tnse_nmf1_xy = json.loads(tnse_nmf1.read())
    tnse_nmf2_xy = json.loads(tnse_nmf2.read())
    tnse_nmf3_xy = json.loads(tnse_nmf3.read())
    tnse_nmf4_xy = json.loads(tnse_nmf4.read())


TNSE_DATAPOINTS = {
    "feature_1": [tnse_mf1_xy, tnse_nmf1_xy],
    "feature_2": [tnse_mf2_xy, tnse_nmf2_xy],
    "feature_3": [tnse_mf3_xy, tnse_nmf3_xy],
    "feature_4": [tnse_mf4_xy, tnse_nmf4_xy],
}


# Reading PCA Plot readings from DB
with (open("pca/murzymes/feature1_murzymes_pca.json", 'r') as pca_mf1, # tnse plot - murzymes - feature - 1
      open("pca/murzymes/feature2_murzymes_pca.json", 'r') as pca_mf2, 
      open("pca/murzymes/feature3_murzymes_pca.json", 'r') as pca_mf3, 
      open("pca/murzymes/feature4_murzymes_pca.json", 'r') as pca_mf4, 
      open("pca/non_murzymes/feature1_non_murzymes_pca.json", 'r') as pca_nmf1, 
      open("pca/non_murzymes/feature2_non_murzymes_pca.json", 'r') as pca_nmf2, 
      open("pca/non_murzymes/feature3_non_murzymes_pca.json", 'r') as pca_nmf3, 
      open("pca/non_murzymes/feature4_non_murzymes_pca.json", 'r') as pca_nmf4, 
      ):
    
    pca_mf1_xy = json.loads(pca_mf1.read())
    pca_mf2_xy = json.loads(pca_mf2.read())
    pca_mf3_xy = json.loads(pca_mf3.read())
    pca_mf4_xy = json.loads(pca_mf4.read())

    pca_nmf1_xy = json.loads(pca_nmf1.read())
    pca_nmf2_xy = json.loads(pca_nmf2.read())
    pca_nmf3_xy = json.loads(pca_nmf3.read())
    pca_nmf4_xy = json.loads(pca_nmf4.read())


PCA_DATAPOINTS = {
    "feature_1": [pca_mf1_xy, pca_nmf1_xy],
    "feature_2": [pca_mf2_xy, pca_nmf2_xy],
    "feature_3": [pca_mf3_xy, pca_nmf3_xy],
    "feature_4": [pca_mf4_xy, pca_nmf4_xy],
}


@app.get("/getAllDatapoints")
def getAllDataPoints() -> dict:
    '''
    Output Structure
    {
    "murzymes": [[key, [v1, v2, v3,v4]], .... ,[keyn, [v1, v2, v3, v4]]],
    "non-murzymes": [[key, [v1, v2, v3,v4]], .... ,[keyn, [v1, v2, v3, v4]]]
    }
    '''
    return {
        "murzymes": [[key, [(list(i) if type(i) == np.ndarray else [0]*50) for i in value]] for key, value in rawMurzymes.items()],
        "non-murzymes": [[k, [(list(j) if type(j) == np.ndarray else [0]*50) for j in v]] for k, v in rawNonMurzymes.items()]
    }


@app.get("/search_datapoint/{search_keyword}")
def searchDataPoint(search_keyword) -> list:
    '''
    datapoints with key values that are similar will be added to a list and sent as response
    Simple key-Word Based Search
    Output: [[k1, [v1, v2, v3, v4], "murzyme"], ... , [kn, [v1, v2, v3, v4], "non-murzyme"]]
    '''
    response = list()
    
    for k_1, v_1 in rawMurzymes.items():
        if search_keyword.casefold() in k_1.casefold():
            response.append([k_1, [(list(vctr) if type(vctr) == np.ndarray else [0]*50) for vctr in v_1], "murzyme"])
    
    for k_2, v_2 in rawNonMurzymes.items():
        if search_keyword.casefold() in k_2.casefold():
            response.append([k_2, [(list(vctr) if type(vctr) == np.ndarray else [0]*50) for vctr in v_2], "non-murzyme"])

    return response


@app.get("/tnse_plot_feature/{feature_number}")
def getTnsePlotForFeatureN(feature_number: int) -> dict:
    """
    feature number can be one among[1, 2, 3, 4]
    response will be all points of x and y of muzymes and non-murzymes of the given plot
    """
    try:
        featureChose = TNSE_DATAPOINTS[F"feature_{feature_number}"]

        murz, non_murz = featureChose[0], featureChose[1]

        murz_x, murz_y = [i[0] for i in murz.values()], [j[1] for j in murz.values()]
        non_murz_x, non_murz_y = [i[0] for i in non_murz.values()], [j[1] for j in non_murz.values()]

        return {
            "murzymes_x": murz_x,
            "murzymes_y": murz_y,
            "non_murzymes_x": non_murz_x,
            "non_murzymes_y": non_murz_y,
        }
    except:
        return {
            "ERROR MESSAGE": "NO SUCH FEATURE FOUND"
        } 




@app.get("/pca_plot_feature/{feature_number}")
def getPcaPlotForFeatureN(feature_number: int) -> dict:
    """
    feature number can be one among[1, 2, 3, 4]
    response will be all points of x and y of muzymes and non-murzymes of the given plot
    """
    try:

        featureChose = PCA_DATAPOINTS[F"feature_{feature_number}"]

        murz, non_murz = featureChose[0], featureChose[1]

        murz_x, murz_y = [i[0] for i in murz.values()], [j[1] for j in murz.values()]
        non_murz_x, non_murz_y = [i[0] for i in non_murz.values()], [j[1] for j in non_murz.values()]

        return {
            "murzymes_x": murz_x,
            "murzymes_y": murz_y,
            "non_murzymes_x": non_murz_x,
            "non_murzymes_y": non_murz_y,
        }
    
    except:
        return {
            "ERROR MESSAGE": "NO SUCH FEATURE FOUND"
        } 


@app.post("/predict_murzyme_or_not")
def predictOutput(input_features: str) -> dict:
    """
    instead of nan's "None" is Expected
    and a normal list of integers or list is expected
    """
    res = extractAndScaleAndPredict(eval(input_features))

    return {"predicted_result": res}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)