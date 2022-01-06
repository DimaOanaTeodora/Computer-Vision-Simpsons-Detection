import os
import cv2
import random
from os import path

class ExtractFaces:
    """
    produce datele de antrenare (fete si non-fete) 
    pentru fiecare director corespunzator fiecarui personaj
    """
    def __init__(self, root_folder, character):
        self.root_folder = root_folder
        self.character = character
        self.train_non_faces = []
        self.train_faces = []
        self.faces_path = './faces/'
        self.non_faces_path = './non_faces/'
        if not path.exists(self.faces_path):
            os.mkdir(self.faces_path) 
        if not path.exists(self.non_faces_path):
            os.mkdir(self.non_faces_path)
        self.detections = {} # detectiile corespunzatoare

        self.load_detections()
        self.load_images()
        self.saveFaces()
        self.saveNonFaces()

    def getNonFaces(self):
        return self.train_non_faces
    def getFaces(self):
        return self.train_faces

    def saveFaces(self):
        for i in range(len(self.train_faces)):
                #TODO: save as npy
                cv2.imwrite(os.path.join(self.faces_path + self.character + '_' + str(i) + '.jpg'),
                            self.train_faces[i])
    def saveNonFaces(self):
        for i in range(len(self.train_non_faces)):
                #TODO: save as npy
                cv2.imwrite(os.path.join(self.non_faces_path + self.character + '_' + str(i) + '.jpg'),
                            self.train_non_faces[i])

    def generateFaces(self, coordinates, image):
        # salveaza fetele
        for x_min, y_min, x_max, y_max in coordinates:
                face = image[y_min : y_max, x_min : x_max]

                # redimensionare la (36,36)
                resized_face = cv2.resize(face, (36, 36))
                # cv2.imshow("resized face", resized_face)
                # cv2.waitKey(0)

                # adauga la datele de training
                self.train_faces.append(resized_face)
    
    def generateNonFaces(self, coordinates, image):
        # augumentare cu imagini care nu sunt fete si au preponderent culoarea galbena
        for i in range(50):
                for j in range(4): # incearca de maxim 4 ori sa gaseasca o non fata
                    # TODO:daca ii dau mai mare ca 4 trebuie sa schimb formula

                    # selectare punct random pentru generare non-fata
                    y = random.randint(0, image.shape[0] - int(36 * (1.5**j) + 1)) #TODO: revin-o aici
                    x = random.randint(0, image.shape[1] - int(36 * (1.5**j) + 1))

                    # nu trebuie sa se intersecteze cu fetele 
                    intersects = False
                    for x_min, y_min, x_max, y_max in coordinates:
                        # TODO: revino si aici
                        intersects  = (x < x_max and y < y_max) and ( (x + int(36 * (1.5**j) + 1)) > x_min and (y + int(36 * (1.5**j) + 1)) > y_min)

                    if intersects is False:
                        # TODO: revino aici
                        non_face = image[y : (y + int(36 * (1.5**j) + 1)), x : (x + int(36 * (1.5**j) + 1))]
                        # TODO revino aici
                        patch_hsv = cv2.cvtColor(non_face, cv2.COLOR_BGR2HSV)
                        yellow_patch = cv2.inRange(patch_hsv, (19, 90, 190), (90, 255, 255))
                        # cv2.imshow("resized non-face", resized_non_face)
                        # cv2.waitKey(0)

                        if yellow_patch.mean() >= 50:
                            non_face = image[y : y + int(36 * (1.5**j) + 1), x : x + int(36 * (1.5**j) + 1)]
                            resized_non_face = cv2.resize(non_face, (36, 36))
                            self.train_non_faces.append(resized_non_face)
                            break

    def load_detections(self):
        """
        citeste imaginile de antrenare si 
        detectiile corespunzatoare
        """

        # ./antrenare/bart.txt
        f = open(self.root_folder + self.character + ".txt")
        for line in f.readlines():
            v = line.split()
            # bart_pic_0016.jpg
            key = self.character + "_" + v[0]
            x_min = int(v[1])
            y_min = int(v[2])
            x_max = int(v[3])
            y_max = int(v[4])

            if key in self.detections.keys():
                self.detections.get(key).append((x_min, y_min, x_max, y_max))
            else:
                self.detections[key] = [(x_min, y_min, x_max, y_max)]

    def load_images(self):  
        # ./antrenare/bart/
        images_path = self.root_folder + self.character + "/"
        for image in os.listdir(images_path):
            # bart_pic_0016.jpg
            key_name = self.character + "_" + image
            # /antrenrae/bart/pic_0016.jpg 
            image_path = images_path + image
            # apeleaza functia de citire a imaginilor si generare a datelor de training
            self.process_image(image_path, key_name)
    
    def process_image(self, image_path, key_name):
        # citeste si salveaza imaginile
        # salveaza fetele si genereaza nonfetele pentru training

        # converteste imaginea la grayscale
        image = cv2.imread(image_path)

        # coordinates = [(xmin, ymin, yxmax, ymax), .....]
        coordinates = self.detections[key_name]
        self.generateNonFaces(coordinates, image)
        self.generateFaces(coordinates, image)

        
t1 = ExtractFaces("./antrenare/", "bart")
t2 = ExtractFaces("./antrenare/", "homer")
t3 = ExtractFaces("./antrenare/", "lisa")
t4 = ExtractFaces("./antrenare/", "marge")

