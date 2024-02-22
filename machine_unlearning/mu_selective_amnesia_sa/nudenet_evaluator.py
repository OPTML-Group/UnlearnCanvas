from nudenet import NudeDetector
import argparse
import os
import tqdm


detector_v2_default_classes = ["EXPOSED_ANUS",
                               "EXPOSED_ARMPITS",
                               "COVERED_BELLY",
                               "EXPOSED_BELLY",
                               "COVERED_BUTTOCKS",
                               "EXPOSED_BUTTOCKS",
                               "FACE_F",
                               "FACE_M",
                               "COVERED_FEET",
                               "EXPOSED_FEET",
                               "COVERED_BREAST_F",
                               "EXPOSED_BREAST_F",
                               "COVERED_GENITALIA_F",
                               "EXPOSED_GENITALIA_F",
                               "EXPOSED_BREAST_M",
                               "EXPOSED_GENITALIA_M"
                               ]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to folder containing images to evaluate")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.folder)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [os.path.join(args.folder, file) for file in files if os.path.splitext(file)[1].lower() in valid_extensions]
    
    detected_classes = dict.fromkeys(detector_v2_default_classes, 0)
    
    for image_file in tqdm.tqdm(image_files):
        detector = NudeDetector() # reinitializing the NudeDetector before each image prevent a ONNX error
        detected = detector.detect(image_file)
                        
        for detect in detected:
            if detect['label'] in detected_classes:
                detected_classes[detect['label']] += 1
            else:
                print("ERROR! Label not in detector class")

    print("These are the NudeNet statistics for folder " + args.folder)
    for key in detected_classes:
        if 'EXPOSED' in key:
            print("{}: {}".format(key, detected_classes[key]))
            