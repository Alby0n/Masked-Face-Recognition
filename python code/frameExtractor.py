def frameExtractor():
    if os.path.exists('images'):
        shutil.rmtree('/content/gdrive/MyDrive/G-Drive/images') #Deleting existing directory to remove garbage files
    cam = cv2.VideoCapture(vid) # video file input
#   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
    try:
        if not os.path.exists('images'):#Creating images folder to store video frames
            os.makedirs('images')
            print("Created")
    except OSError:
        print('Error: Creating directory of images') #Error message
    currentframe = 0
    while (True):
        ret, frame = cam.read()
        if ret:
            frameExtractor(frame)
            # name = './images/frame' + str(currentframe) + '.jpg' #name of the file to be saved
            # print('Creating...' + name) #Creating frame by frame
            # cv2.imwrite(name, frame) #saving the frame to the folder images
            currentframe += 1 #incrementing the frame number
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
