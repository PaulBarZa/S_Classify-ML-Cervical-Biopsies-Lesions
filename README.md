# Cytomine Software - Classify-ML-Cervical-Biopsies-Lesions

* **Summary:** It uses trained CNNs (EfficientNet-BO) model to classify microscopic slides of uterine cervical according to the most severe category of epithelial lesion present in the sample.

* **Based on:** 
    
    * Third place algo of the [TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/).

    * Author : [kbrodt](https://github.com/kbrodt). 

    * Code repository : [here](https://github.com/kbrodt/competition-cervical-biopsy).


* **Parameters:** 
  * *cytomine_host*, *cytomine_public_key*, *cytomine_private_key*,*cytomine_id_project*, *cytomine_id_images* and *cytomine_id_software* are parameters needed for Cytomine external app. They will allow the app to be run on the Cytomine instance (determined with its host), connect and communicate with the instance (with the key pair). An app is always run into a project (*cytomine_id_project*) and to be run, the app must be previously declared to the plateform (*cytomine_id_software*). Note that if *cytomine_id_images* is not specified, the segmentation will be applied on all images from the project.

* **Output:**
    * A file mapping selected images name with predicted classe.
    * Possible output classes :
        * Benign
        * Low malignant potential
        * High malignant potential
        * Invasive cancer