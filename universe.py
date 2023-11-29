from concurrent.futures import ThreadPoolExecutor
import os
import csv
import requests
from tqdm import tqdm
from roboflow import Roboflow
import threading

def process_image(filename):
    img_path = os.path.join(image_folder, filename)
    for model, confidence_level in projects_and_models:
        try:
            result = model.predict(img_path, confidence=confidence_level, overlap=55).json()
            if result["predictions"]:
                for pred in result["predictions"]:
                    if pred["class"] in target_classes:
                        xmin = pred['x'] - pred['width'] / 2
                        xmax = pred['x'] + pred['width'] / 2
                        ymin = pred['y'] - pred['height'] / 2
                        ymax = pred['y'] + pred['height'] / 2

                        pred_data = {
                            "filename": filename,
                            "width": pred["width"],
                            "height": pred["height"],
                            "class": pred["class"],
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax
                        }
                        with lock:
                            with open(csv_file, 'a') as f:
                                writer = csv.DictWriter(f, fieldnames=csv_columns)
                                writer.writerow(pred_data)
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection Error: {errc}")


rf = Roboflow(api_key="9E1nYlDzUffWDBatWpTW")

projects_and_models = [
    (rf.workspace().project("forklift-odgis").version(1).model, 70),
    (rf.workspace().project("fire-per").version(1).model, 70),
    (rf.workspace().project("container-yara2").version(2).model, 70),
    (rf.workspace().project("qr-y28gs").version(1).model, 70),
    (rf.workspace().project("barcodes_mironet").version(2).model, 80),
    (rf.workspace().project("ainzyo-anpr-wkwwc").version(1).model, 80),
    (rf.workspace().project("detect-and-recognize-traffic-sign").version(1).model, 80),
    (rf.workspace().project("highway-object-detection").version(38).model, 80),
    (rf.workspace().project("box-object-detection").version(2).model, 80),
    (rf.workspace().project("safety_ppe").version(1).model, 75),
    (rf.workspace().project("site-construction-safety").version(1).model, 65),
    (rf.workspace().project("traffic-light-detection-xti1a").version(1).model, 80),
    (rf.workspace().project("cone-detection-ukhe8").version(1).model, 65),
    (rf.workspace().project("container-number-detection").version(1).model, 65),
    (rf.workspace().project("container-serials").version(1).model, 80),
    (rf.workspace().project("traffic-kdtnh").version(1).model, 80),
    (rf.workspace().project("vehicle-classification-sgcum").version(3).model, 80),
    (rf.workspace().project("mask_rcnn-73vt5").version(4).model, 80),
    (rf.workspace().project("mamgistics").version(1).model, 80),
    (rf.workspace().project("people-and-ladders").version(4).model, 80),
    (rf.workspace().project("ladder-bucu7").version(1).model, 80),
    (rf.workspace().project("personal-protective-equipment-combined-model").version(4).model, 80),
]


csv_file = "predictions.csv"
csv_columns = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
target_classes = ["person", "forklift", "fire", "smoke", 'container', 'qr_code', "barcode", "licence-plate", "Traffic-Sign", "car", "truck", 'Boxes', "Glove", "Helmet", "Person", "Gloves", "Vest", "traffic light", "cone", "container-number", "container_number", "Container-number", "van", "pallet", "boxes", "Person", "Ladder", "ladder", "Hardhat", "Safety Cone", "Safety Vest"]  # Fill this with your target classes

if not os.path.exists(csv_file):
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

image_folder = "./images"
lock = threading.Lock()

with ThreadPoolExecutor(max_workers=10) as executor:
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    list(tqdm(executor.map(process_image, image_files), total=len(image_files)))
