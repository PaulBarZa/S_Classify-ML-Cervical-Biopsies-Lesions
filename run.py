import os
import logging
import shutil
from pathlib import Path

from cytomine import CytomineJob
from cytomine.models import *
from predict import perform_inference

logging.basicConfig()
logger = logging.getLogger("run")
logger.setLevel(logging.INFO)

CLASS_MESSAGES = ["Benign", "Low malignant potential", "High malignant potential", "Invasive cancer"]
ERROR_CLASS_MESSAGE = "Invalid class prediction has been done. Predicted value should be between 0 and 3."
TEMP_IMAGES_DIR = "images"
OUTPUT_FILENAME = "prediction.txt"
OUTPUT_COMMENT = "Prediction file"

def filter_images(images, images_id):
    filtered_images = []
    for image in images:
        if str(image.id) in images_id:
            filtered_images.append(image)

    return filtered_images

def get_prediction_label(prediction):
    if(prediction >= 0 & prediction <= 3):
        return CLASS_MESSAGES[prediction]
    else:
        return ERROR_CLASS_MESSAGE

def main(argv):
    with CytomineJob.from_cli(argv) as cj:  

        logger.info("Running job with parameter: %s", cj.parameters)

        # Create images directory from working path
        root_path = str(Path.home())
        working_path = os.path.join(root_path, TEMP_IMAGES_DIR)
        os.makedirs(working_path, exist_ok=True)
        
        # Fetch project images
        imageCollection = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        images_id = cj.parameters.cytomine_id_images

        # Filter images by id
        filtered_images = []
        if images_id != 'all':
            filtered_images = filter_images(imageCollection, images_id.split(','))
        else:
            filtered_images = imageCollection

        images_filename = []

        cj.job.update(progress=1, statusComment="Preparing execution (creating folders and downloading image(s)/ROI(s)).")

        logger.info("Downloading images")
        cj.job.update(progress=5, statusComment="Downloading images in: {}".format(working_path))
        # If a ROI term is specified
        if(cj.parameters.cytomine_id_roi_term):
            for image in filtered_images:
                # For each image fetch ROI annotations
                roi_annotations = AnnotationCollection(
                    terms=[cj.parameters.cytomine_id_roi_term],
                    project=cj.parameters.cytomine_id_project,
                    image=str(image.id),
                    showWKT = True,
                    includeAlgo=True, 
                )
                roi_annotations.fetch()

                for roi in roi_annotations:
                    # Download annotations as png file
                    filename = 'roi_' + str(roi.id) + '_in_image_' + image.filename +'.png'
                    logger.info("Download {}".format(filename))
                    is_algo = User().fetch(roi.user).algo
                    roi.dump(dest_pattern=os.path.join(working_path, filename), mask = True, alpha = not is_algo)
                    # Add the filename to our list
                    images_filename.append(filename)

        # If a ROI term is specified
        else:
            for image in filtered_images:
                filename = image.filename
                logger.info("Download {}".format(filename))
                image.download(os.path.join(working_path, filename))
                images_filename.append(filename)

        progress = 10
        progress_message = "{} image(s) downloaded.".format(len(filtered_images))
        cj.job.update(progress=progress, statusComment=progress_message)
        logger.info(progress_message)

        logger.info("Starting inference")
        predictions = perform_inference(cj, images_filename, working_path, progress)

        logger.info("Sending result to UI")
        output_filename = OUTPUT_FILENAME
        output_path = os.path.join(working_path, output_filename)

        # Writing result
        f= open(output_path,"w+")
        for (i, filename) in enumerate(images_filename):
            f.write("Prediction for image {}: {}.\r\n".format(filename, get_prediction_label(predictions[i])))
        f.close() 

        # Sending result to UI
        job_data = JobData(cj.job.id, OUTPUT_COMMENT, output_filename).save()
        job_data.upload(output_path)

        shutil.rmtree(working_path, ignore_errors=True)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])