## Murzyme Classification with Machine Learning (SVM-RBF)

This repository implements a machine learning model using Support Vector Machines (SVM) with RBF kernel for classifying murzymes and non-murzymes. It also provides FastAPI endpoints for data access, dimensionality reduction visualization (t-SNE and PCA), and prediction.

### Features

* **Machine Learning Model:** Leverages SVM-RBF for murzyme classification.
* **FastAPI Endpoints:**
    * GET: Retrieve all data points in the dataset.
    * GET: Search for a data point by ID.
    * GET: Get x-y coordinates for t-SNE plot of specified data points.
    * GET: Get x-y coordinates for PCA plot of specified data points.
    * POST: Predict murzyme class for user-provided features.
* **Preprocessing Pipeline:** Implements feature extraction, engineering, imputation, overbalancing, and scaling.


### Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/murzyme-classification.git
   ```

2. **Install dependencies:**

   ```bash
   pip install requirements.txt
   ```

3. **Run the application:**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   (Replace `8000` with your desired port number if needed)

### API Endpoints

**1. Get All Data Points (GET /data)**

* Returns a JSON object containing all data points in the dataset.

**2. Search Data Point by ID (GET /data/{id})**

* Replace `{id}` with the desired data point ID.
* Returns a JSON object containing the data point with the specified ID.

**3. Get t-SNE Coordinates (GET /data/tsne/{data_ids})**

* Replace `{data_ids}` with a comma-separated list of data point IDs.
* Returns a JSON object containing x-y coordinates for a t-SNE plot of the specified data points.

**4. Get PCA Coordinates (GET /data/pca/{data_ids})**

* Replace `{data_ids}` with a comma-separated list of data point IDs.
* Returns a JSON object containing x-y coordinates for a PCA plot of the specified data points.

**5. Predict Murzyme Class (POST /predict)**

* Send a JSON object containing the data point features in the request body.
* Returns a JSON object with the predicted murzyme class (murzyme or non-murzyme).


### Dataset

The dataset used for training and testing the model is not included in this repository due to potential size or sensitivity constraints. You can replace the data loading logic in `main.py` to point to your own dataset. 

### Model Preprocessing

The preprocessing steps used on the dataset, including feature extraction, engineering, imputation, overbalancing, and scaling, are implemented in the `preprocessing.py` script.


### Contribution

We welcome contributions to this project! Please refer to the CONTRIBUTING.md file for guidelines.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
