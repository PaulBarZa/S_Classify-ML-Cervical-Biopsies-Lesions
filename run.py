
import os
import logging
import shutil
from pathlib import Path

import efficientnet_pytorch
from cytomine import CytomineJob
from cytomine.models import *
from predict import perform_inference

def get_model(model_name, num_classes):
    return efficientnet_pytorch.EfficientNet.from_pretrained(
        model_name,
        num_classes=num_classes,
    ).cuda().eval()

def filter_images(images, images_id):
    filtered_images = []
    for image in images:
        if str(image.id) in images_id:
            filtered_images.append(image)

    return filtered_images

def main(argv):
    with CytomineJob.from_cli(argv) as cj:   

        logging.info("Running job with parameter: %s", cj.parameters)

        # Fetch project images
        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        images_id = cj.parameters.cytomine_id_images

        # Filter images by id
        list_images = []
        if images_id != 'all':
            list_images = filter_images(images, images_id.split(','))
        else:
            list_images = images

        cj.job.update(progress=1, statusComment="Preparing execution (creating folders and downloading image(s)).")
        
        # Create images directory from working path
        root_path = str(Path.home())
        working_path = os.path.join(root_path, "images")
        os.makedirs(working_path, exist_ok=True)

        # And download them
        logging.info("Downloading images")
        cj.job.update(progress=5, statusComment="Downloading images in: {}".format(working_path))
        for image in list_images:
            image.download(os.path.join(working_path, image.filename))

        # Fetching model
        model_name = 'efficientnet-b0'
        num_classes = 3
        progress_message = "Fetching model: {} with {} classes.".format(model_name, num_classes)

        cj.job.update(progress=10, statusComment=progress_message)
        logging.info(progress_message)

        model = get_model(model_name, num_classes)

        # Inference
        logging.info("Starting inference")
        predictions = perform_inference(cj, list_images, model, working_path)

        logging.info("Sending result to UI")
        output_filename = "prediction.txt"
        output_path = os.path.join(working_path, output_filename)

        # Write result
        f= open(output_path,"w+")
        for (i, image) in enumerate(list_images):
            f.write("Prediction for image {}: {}.\r\n".format(image.filename, predictions[i]))
        f.close() 

        # Send result to UI
        job_data = JobData(cj.job.id, "Generated File", output_filename).save()
        job_data.upload(output_path)

        shutil.rmtree(working_path, ignore_errors=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])