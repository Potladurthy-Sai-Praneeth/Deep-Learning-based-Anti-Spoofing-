import face_recognition
import time
from pickle import load
import os
import serial
from classifier import CDC, ClassifierUCDCN
from generate_face_embeddings import generate_embeddings
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

parser = argparse.ArgumentParser(description='Pass user arguments for authentication')

# When this argument is passed along with the path containing the user images the face embeddings are generated.
parser.add_argument('--generate', type=str, default=None, help='Command to generate embeddings')
# parser.add_argument('file_path', type=str, default=None, help='Please provide the path to the images directory')


class AuthenticateUser():
    def __init__(self, face_encdoings_folder,models_folder,video_input=0):
        '''
        This class is used to authenticate the user using face recognition and anti-spoofing techniques.
        Args :
            face_encdoings_folder : str
                The path to the folder containing the face embeddings.
            models_folder : str
                The path to the folder containing the models.
            video_input : int (default : 0) - Optional
                The video input source. Default is the webcam.
        '''
        assert face_encdoings_folder, 'The path to the face embeddings folder is not provided. Please provide the path.'
        assert models_folder, 'The path to the models folder is not provided. Please provide the path.'
        assert os.path.exists(face_encdoings_folder), 'The face embeddings folder does not exist. Please provide the correct path.'
        assert os.path.exists(models_folder), 'The models folder does not exist. Please provide the correct path.'

        if not os.listdir(face_encdoings_folder):
            raise FileNotFoundError('There are no embeddings in the folder')
        
        if not os.listdir(models_folder):
            raise FileNotFoundError('There are no model weights inside the folder')

        self.known_faces = []
        self.known_names = []
        self.face_encdoings_folder = face_encdoings_folder
        self.models_folder = models_folder

        for i in os.listdir(face_encdoings_folder):
            self.known_names.append(i.split('-')[1].split('.')[0])
            self.known_faces.append(np.load(os.path.join(face_encdoings_folder,i)))
        
        print('Finished loading face embeddings')

        self.depth_model_pth_file = None
        self.classifier_pth_file = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in os.listdir(models_folder):
            if i.endswith('.pth') or i.endswith('.pt'):
                if 'depth' in i:
                    self.depth_model_pth_file = os.path.join(models_folder,i)
                elif 'classifier' in i:
                    self.classifier_pth_file = os.path.join(models_folder,i)     

        self.classifier = ClassifierUCDCN(depth_map_path=self.depth_model_pth_file,device=self.device,dropout=0.5,load_depth_model=True).to(self.device)
        self.classifier.load_state_dict(torch.load(self.classifier_pth_file,map_location=self.device))
        self.classifier.eval()

        print('Finished loading models')

        self.video_input = video_input
        self.video_capture = cv2.VideoCapture(video_input)

        self.is_authenticated = False
    
    def get_video_capture(self):
        '''
        This function returns the video capture object.
        '''
        return self.video_capture
    

    def preprocess_image(self, image, target_size=(252, 252)):
        '''
            This function preprocesses the image for the classifier.
            Args :
                image : np.array
                    The image to be preprocessed.
                target_size : tuple (default : (252, 252)) - Optional
                    The target size of the image.
            Returns :
                image : torch.tensor
                    The preprocessed image.
        '''
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Adjust these values to your needs
            #                      std=[0.229, 0.224, 0.225])   # Adjust these values to your needs
        ])
        image = transform(image)
        
        # Check if the image is single channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image.unsqueeze(0).to(self.device)
    
    def authenticate(self):
        '''
        This function authenticates the user using face recognition and anti-spoofing techniques.
        '''
        while True:
            ret, frame = self.video_capture.read()
            if ret:
                # Find all the faces in the frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                # Convert the frame from BGR to RGB
                for face_encoding in face_encodings:
                    # Convert the frame from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Preprocess the frame as a tensor for the classifier
                    preprocessed_frame = self.preprocess_image(frame_rgb)

                    # Make predictions on the preprocessed frame
                    with torch.no_grad():
                        output,depth_map = self.classifier(preprocessed_frame)
                    
                    # Get the prediction
                    prediction = torch.argmax(output).item()
                    print(f"Predicted is {prediction} class: {'Real' if prediction == 1 else 'Spoof'}")

                    depth_map_display = cv2.applyColorMap(depth_map.squeeze(0).squeeze(0).cpu().numpy().astype('uint8'), cv2.COLORMAP_JET)
                    # Display the resulting frame
                    cv2.imshow('Live Camera Feed', frame)
                    # Display the depth map
                    cv2.imshow('Depth Map', depth_map_display)

                    if prediction == 1:
                        # If the prediction is real, proceed with face recognition
                        print('Live User, Proceeding for face recognition')

                        # Compare the faces with the known faces
                        results = face_recognition.compare_faces(self.known_faces, face_encoding)
                        get_name=None
                        for i in range(len(results)):
                            if results[i]:
                                get_name = self.known_names[i]
                                if get_name is not None:
                                    self.is_authenticated = True
                                    break
                                else:
                                    self.is_authenticated = False           
                        if self.is_authenticated:
                            print(f'Authenticated as {get_name}')
                            # Perform any necessary operations here

                            # Add a delay to prevent multiple authentications and reset the authentication
                            self.is_authenticated = False           
                        else:
                            print(f'Person is not a registered user')
                            self.is_authenticated = False
                    else:
                        print('Spoof detected')
                        self.is_authenticated = False
   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.generate is not None:
        if os.path.exists(args.generate):
            generate_embeddings(args.generate)
        else:
            raise FileNotFoundError('The path to the images directory is incorrect or not provided. Please provide the correct path.')
    else:
        if os.path.exists('face_embeddings'):
            auth = AuthenticateUser(os.path.join(os.getcwd(),'face_embeddings'),os.path.join(os.getcwd(),'models'))
            auth.authenticate()    
        else:
            raise FileNotFoundError('The face embeddings are not found. Please generate the embeddings first.')