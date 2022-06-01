from tensorflow.keras.models import model_from_json
def faceRecognizer():
    json_file = open('../MyDrive/G-Drive/model_data/model.json', 'r') #loading the json file
    loaded_model_json = json_file.read() #loading the json file
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('../G-Drive/model_data/model.h5')
    FRmodel = model
    def img_to_encoding(image_path, model): #Function to convert the image to the 128 length vector encoding
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160)) #Resizing the image to 160x160
    
        img = np.around(np.array(img) / 255.0, decimals=12) #Normalizing the image
        x_train = np.expand_dims(img, axis=0) #Expanding the image to a 4D array
        embedding = model.predict_on_batch(x_train) #Predicting the 128 length vector encoding
        return embedding / np.linalg.norm(embedding, ord=2) #Normalizing the 128 length vector encoding

    database = {}

    def who_is_it(image_path, database, model):
        encoding = img_to_encoding(image_path,model) #Converting the image to the 128 length vector encoding

        min_dist = 100 #Setting the minimum distance to 100

        for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. 
            dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. 
            if dist < min_dist:
                min_dist = dist
                identity = name
    
    
        if min_dist > 1.2:
        #print("Not in the database.")
            pass
        else:
            pass
        #print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        #print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
        return min_dist, identity