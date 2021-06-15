# bento_packer.py

# import the IrisClassifier class defined above
try:
    from bento_service import IrisClassifier
    from train import clf
except:
    from bentoml_deploy.bento_service import IrisClassifier
    from bentoml_deploy.train import clf

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
