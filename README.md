# iris-classification-bentoml
Iris Classification problem deployed using BentoML

## Prerequisites

Install dependencies

```sh
pip install -r requirements.txt
```

## Folder structure

```root
bento_deploy/
├── bento_packer.py  # responsible for packing BentoService
├── bento_service.py # BentoService definition
├── model.py         # DL Model definitions
├── train.py         # OPTIONAL: training scripts
└── requirements.txt
```

- train.py

We then need to prepare a trained model before serving with BentoML. Train a classifier model with Scikit-Learn on the Iris data set.

```python
# train.py
from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)
```

- bento_service.py

Model serving with BentoML comes after a model is trained. The first step is creating a prediction service class, which defines the models required and the inference APIs which contains the serving logic code. Here is a minimal prediction service created for serving the iris classifier model trained above, which is saved under `bento_service.py`.

```python
# bento_service.py
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)
```

Firstly, the `@artifact(...)` here defines the required trained models to be packed with this prediction service. BentoML model artifacts are pre-built wrappers for persisting, loading and running a trained model. This example uses the `SklearnModelArtifact` for the scikit-learn framework. BentoML also provide artifact class for other ML frameworks, including `PytorchModelArtifact`, `KerasModelArtifact`, and `XgboostModelArtifact` etc.

The `@env` decorator specifies the dependencies and environment settings required for this prediction service. It allows BentoML to reproduce the exact same environment when moving the model and related code to production. With the `infer_pip_packages=True` flag, BentoML will automatically find all the PyPI packages that are used by the prediction service code and pins their versions.

The `@api` decorator defines an inference API, which is the entry point for accessing the prediction service. The `input=DataframeInput()` means this inference API callback function defined by the user, is expecting a pandas.DataFrame object as its input.

When the batch flag is set to `True`, an inference APIs is suppose to accept a list of inputs and return a list of results. In the case of DataframeInput, each row of the dataframe is mapping to one prediction request received from the client. BentoML will convert HTTP JSON requests into pandas.DataFrame object before passing it to the user-defined inference API function.

This design allows BentoML to group API requests into small batches while serving online traffic. Comparing to a regular flask or FastAPI based model server, this can largely increase the overall throughput of the API server.

Besides `DataframeInput`, BentoML also supports API input types such as `JsonInput`, `ImageInput`, `FileInput` and more. `DataframeInput` and `TfTensorInput` only support inference API with `batch=True`, while other input adapters support either batch or single-item API.

- bento_packer.py

The following code packages the trained model with the prediction service class IrisClassifier defined above, and then saves the IrisClassifier instance to disk in the BentoML format for distribution and deployment, under `bento_packer.py`.

```python
# bento_packer.py

# import the IrisClassifier class defined above
from bento_service import IrisClassifier

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
```

BentoML stores all packaged model files under the `~/bentoml/repository/{service_name}/{service_version}` directory by default. The BentoML packaged model format contains all the code, files, and configs required to run and deploy the model.

BentoML also comes with a model management component called `YataiService`, which provides a central hub for teams to manage and access packaged models via Web UI and API.

![imag](https://docs.bentoml.org/en/latest/_images/yatai-service-web-ui-repository.png)

![imag](https://docs.bentoml.org/en/latest/_images/yatai-service-web-ui-repository-detail.png)

Launch Yatai server locally with docker and view your local repository of BentoML packaged models:

```sh
docker run \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/bentoml:/bentoml \
  -p 3000:3000 \
  -p 50051:50051 \
  bentoml/yatai-service:latest
```

## Model Serving via REST API

To start a REST API model server locally with the `IrisClassifier` saved above, use the bentoml serve command followed by service name and version tag:

```sh
bentoml serve IrisClassifier:latest
```

The IrisClassifier model is now served at `localhost:5000`. The web UI is served at another URL. 
Use curl command to send a prediction request:

```sh
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://localhost:5000/predict
```

Or with python and request library:

```python
import requests
response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
print(response.text)
```

The BentoML API server also provides a simple web UI dashboard. Go to `http://localhost:5000` in the browser and use the Web UI to send prediction request:

![imag](https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png)

## Launch inference job from CLI

The BentoML CLI supports loading and running a packaged model from CLI. With the DataframeInput adapter, the CLI command supports reading input Dataframe data directly from CLI arguments and local files:

```sh
bentoml run IrisClassifier:latest predict --input '[[5.1, 3.5, 1.4, 0.2]]'
```

```sh
bentoml run IrisClassifier:latest predict --input-file './iris_data.csv'
```

## Containerize Model API Server

One common way of distributing this model API server for production deployment, is via Docker containers. And BentoML provides a convenient way to do that.

If you already have docker configured, run the following command to build a docker container image for serving the IrisClassifier prediction service created above:

```sh
bentoml containerize IrisClassifier:latest -t iris-classifier
```

Start a container with the docker image built from the previous step:

```sh
docker run -p 5000:5000 iris-classifier:latest --workers=2
```