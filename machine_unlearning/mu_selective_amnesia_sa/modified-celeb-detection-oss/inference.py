import os
import argparse
import moviepy.editor as mov_editor

from dotenv import load_dotenv
from skimage import io
from pprint import pprint

from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector

import numpy as np
from tqdm import tqdm

def process_gif(path):
    gif = mov_editor.VideoFileClip(path)
    selected_frames = evenly_spaced_sampling(list(gif.iter_frames()), gif_frames)
    face_images_by_frames = face_detector.perform_bulk(selected_frames, range(len(selected_frames)))
    face_images = []
    for frame_faces in face_images_by_frames.values():
        face_images.extend([preprocess_image(image, image_size) for image, _ in frame_faces])
    return face_recognizer.perform(face_images)


def process_image(path):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    return face_recognizer.perform(face_images)


if __name__ == '__main__':
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description='Inference script for Giphy Celebrity Classifier model')
    parser.add_argument('--image_folder', type=str, help='path or link to the image folder', default=None)
    parser.add_argument('--celebrity', type=str, choices=['brad pitt', 'angelina jolie'], required=True)
    # group.add_argument('--gif_path', type=str, help='path or link to the gif', default=None)
    args = parser.parse_args()

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

    model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))
    face_detector = FaceDetector(
        os.getenv('APP_DATA_DIR'),
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA') == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv('APP_DATA_DIR'),
        use_cuda=os.getenv('USE_CUDA') == "true"
    )

    p_celebrity = []
    p_celebrity_given_face = []
    file_name_and_probability = []
    
    if args.celebrity == 'brad pitt':
        i = 281
    elif args.celebrity == 'angelina jolie':
        i = 124
        
    entropy = []
    n_no_faces = 0
    
    # Iterate through all files in the given folder path
    for file in tqdm(os.listdir(args.image_folder)):
        entropy_temp = 0
        file_path = os.path.join(args.image_folder, file)  # Obtain the full file path
        if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
            predictions = process_image(file_path)
            
            if len(predictions) == 0:
                n_no_faces += 1
                p_celebrity.append(0.)
            
            else:
                
                # if there are n faces in an image, len(predictions) == n
                # after sorting predictions[0] corresponds to face with highest probability of target celebrity in an image
                predictions = sorted(predictions, reverse=True, key = lambda x: x[:][0][i][1])
                # print(file)
                # if len(predictions) > 1:
                #     for j in range(len(predictions)):
                #         print(predictions[j][0][i][1])
                
                for pred in predictions[0][0]:
                    entropy_temp -= pred[1] * np.log(pred[1]) # pred[i] is the probability of each of the 2706 celebrity corresponding to a given face
                entropy.append(entropy_temp)
                
                p_celebrity_given_face.append(predictions[0][0][i][1])
                p_celebrity.append(predictions[0][0][i][1])
                file_name_and_probability.append((predictions[0][0][i][1], file))
    
    #np.save("p_celebrity_given_face.npy", np.array(p_celebrity_given_face))
    print(f"For image folder {args.image_folder}")
    print(f"proportion of images with no faces detected: {n_no_faces/len(os.listdir(args.image_folder))}")
    #print(f"avg prob of {args.celebrity}: {np.mean(p_celebrity)}+-{np.std(p_celebrity)}")
    print(f"avg prob of {args.celebrity} given face: {np.mean(p_celebrity_given_face)} +- {np.std(p_celebrity_given_face)}")
    print(f"avg entropy given face: {np.mean(entropy)}")
    
    file_name_and_probability = sorted(file_name_and_probability, reverse=True, key = lambda x:x[0])
    print("Top images with high probability: ")
    print(file_name_and_probability[:10])
    
    