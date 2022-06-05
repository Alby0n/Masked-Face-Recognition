from faceRecognizer import who_is_it
import cv2
def faceDetector(image):
    base_dir = os.path.dirname(os.getcwd())#setting up the base dir to access files
    prototxt_path = os.path.join(base_dir + 'model_data/deploy.prototxt') #The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
    caffemodel_path = os.path.join(base_dir + 'model_data/weights.caffemodel') #The .caffemodel file which contains the weights for the actual layer

  # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path) #loading the model
  
  # Create directory 'faces' if it does not exist
    if not os.path.exists('faces'):
        print("New directory created")
        os.makedirs('faces')

  # Loop through all images and strip out faces
    count = 0 #counting the number of faces [CODE TO BE REVISED]
    Flg=False #flag to check if the recognition is over
    list5=[] 
    flag=False
    gflag=False
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    try:
    # Identify each face
      for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

              # If confidence > 0.15, save it as a separate file . It is set to 0.15 is to detect faces with masks but the best practice is to use 0.5
        if (confidence > 0.15):
          count += 1
          frame = image[startY:endY, startX:endX] #cropping the face
                  
          nope,id=who_is_it(frame, database, FRmodel)#Passing detected faces for recognition
                    

          if(float(nope)<1.2): #After the confidence (the distance) is good, print the person recognised
            realMen="/content/gdrive/MyDrive/G-Drive/Temp Folder/"+id
            realMen=cv2.imread(realMen)
            Flg=True
            cv2.imwrite(base_dir + 'faces/'   + "gen.jpg", frame)#Detected face
            return image,frame,realMen,id
                      
    except:
      pass  
    if Flg==False:
        ss="None found in the database"#When not found in the database
        fakeim=cv2.imread("/content/gdrive/MyDrive/G-Drive/1200px-No_Cross.svg.png")#Can customise the output NOT FOUND image by changing path
        return fakeim,fakeim,fakeim,ss
    # for file in os.listdir(base_dir + 'images'):
    #     file_name, file_extension = os.path.splitext(file)
    #     if (file_extension in ['.png','.jpg']):
          
    #         image = cv2.imread(base_dir + 'images/' + file) #pass the frame here and remove the loop

    #         (h, w) = image.shape[:2]
    #         blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #         model.setInput(blob)
    #         detections = model.forward()
    #         try:
    #       # Identify each face
    #             for i in range(0, detections.shape[2]):
    #                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #box is the coordinates of the face
    #                 (startX, startY, endX, endY) = box.astype("int")

    #                 confidence = detections[0, 0, i, 2]

    #           # If confidence > 0.15, save it as a separate file . It is set to 0.15 is to detect faces with masks but the best practice is to use 0.5
    #                 if (confidence > 0.15):
    #                     count += 1
    #                     frame = image[startY:endY, startX:endX] #cropping the face
    #                     cv2.imwrite(base_dir + 'faces/'   + "gen.jpg", frame)#Detected face
                  
    #                     nope,id=who_is_it(base_dir + 'faces/'   + "gen.jpg", database, FRmodel)#Passing detected faces for recognition
                    

    #                     if(float(nope)<1.2): #After the confidence (the distance) is good, print the person recognised
    #                         realMen="/content/gdrive/MyDrive/G-Drive/Temp Folder/"+id
    #                         realMen=cv2.imread(realMen)
    #                         Flg=True
    #                         return image,frame,realMen,id
                      
    #         except:
    #             pass  
    # if Flg==False:
    #     ss="None found in the database"#When not found in the database
    #     fakeim=cv2.imread("/content/gdrive/MyDrive/G-Drive/1200px-No_Cross.svg.png")#Can customise the output NOT FOUND image by changing path
    #     return fakeim,fakeim,fakeim,ss
