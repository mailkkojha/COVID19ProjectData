import tkinter as tk
from PIL import Image, ImageTk,ImageOps
from tkinter.filedialog import askopenfilename
import cv2
import imutils
import numpy as np
import tensorflow.keras


classes = {
            1:'COVID',
            2:'LUNG OPACITY',
            3:'NORMAL',
            4:'VIRAL PNEUMONIA',
           }

window = tk.Tk()
window.title("Covid-19_Detection")

window.geometry('1300x650')
window.configure(background='snow')

load = Image.open('image_1.jpeg')
render = ImageTk.PhotoImage(load)
img = tk.Label(window, image = render)
img.place(x=0,y=0)

message = tk.Label(window, text="Covid-19 Detection Using X-Ray Images", bg="snow", fg="black", width=60,
                   height=2, font=('times', 30, 'italic bold '))
message.place(x=0, y=0)

window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file='ic.png'))

#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('covid_model.h5')

def exit():
        window.destroy()

def clear():
      rst.destroy()

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

def model_predict():
    img = cv2.imread(path)
    cv2.imwrite("in.jpg",img)
        # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224,3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("in.jpg")

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    idx = np.argmax(prediction)
    final_prediction = classes[idx+1]        
        
    global rst
    if (final_prediction == 'COVID'):
            R = "COVID-19 - TRUE \n \n LUNG OPACITY - FALSE \n\n  NORMAL - FALSE \n \n VIRAL PNEUMONIA - FALSE \n\n\n\n "
            R2 = "This X-Ray image of the lung is predicted to have \n\n been infected with COVID-19"
            rst = tk.Label(text='REPORT \n\n '+  R + R2 , background="snow",fg="Black", font=("", 14,'bold'))
            rst.place(x=810,y=150)
    elif (final_prediction == 'LUNG OPACITY'):
            R = "COVID-19 - FALSE \n \n LUNG OPACITY - TRUE \n\n  NORMAL - FALSE \n \n VIRAL PNEUMONIA - FALSE \n\n\n\n "
            R2 = "This X-Ray image of the lung is predicted to have \n\n been infected with LUNG OPACITY"
            rst = tk.Label(text='REPORT \n\n '+  R + R2 , background="snow",fg="Black", font=("", 14,'bold'))
            rst.place(x=810,y=150)
    elif (final_prediction == 'NORMAL'):
            R = "COVID-19 - FALSE \n \n LUNG OPACITY - FALSE \n\n  NORMAL - TRUE \n \n VIRAL PNEUMONIA - FALSE \n\n\n\n "
            R2 = "This X-Ray image of the lung is predicted to be \n \n NORMAL"
            rst = tk.Label(text='REPORT \n\n '+  R + R2 , background="snow",fg="Black", font=("", 14,'bold'))
            rst.place(x=810,y=150)
    elif (final_prediction == 'VIRAL PNEUMONIA'):
            R = "COVID-19 - FALSE \n \n LUNG OPACITY - FALSE \n\n  NORMAL - FALSE \n \n VIRAL PNEUMONIA - TRUE \n\n\n\n "
            R2 = "This X-Ray image of the lung is predicted to have \n\n been infected with VIRAL PNEUMONIA"
            rst = tk.Label(text='REPORT \n\n '+  R + R2 , background="snow",fg="Black", font=("", 14,'bold'))
            rst.place(x=810,y=150)
            
##    rst = tk.Label(text='RESULT\n\n '+  final_prediction.upper(), background="snow",fg="Black", font=("", 20,'bold'))
##    rst.place(x=810,y=200)
    clrWindow = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    clrWindow.place(x=90, y=390)
    
def openphoto():
    global path
    path=askopenfilename(filetypes=[("Image File",'')])
    im = cv2.imread(path)
    cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=350)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(window,image = tkimage, height="350", width="350")
    myvar.image = tkimage
    myvar.place(x=390,y=150)
    preImg = tk.Button(window, text="Predict",fg="black",command=model_predict ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    preImg.place(x=90, y=270)

takeImg = tk.Button(window, text="Select Image",command=openphoto,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=150)

quitWindow = tk.Button(window, text="Quit", command=on_closing  ,fg="white"  ,bg="Red"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1000, y=550)

window.mainloop()
