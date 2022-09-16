
#Backend code
#importing all libraries
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tensorflow.keras.utils import load_img, img_to_array


#loading the datase
fire = glob.glob('1/*.jpg')
Nonfire = glob.glob('0/*.jpg')

#merging the two datasets
fire_list = []
nonfire_list = []
for x in fire:
  fire_list.append([x,"Fire"])
for x in Nonfire:
  nonfire_list.append([x,"No_fire"])

dataset = fire_list+nonfire_list


#Creating a Dataframe
df = pd.DataFrame(dataset,columns = ['image','label'])

df = df.sample(frac=1).reset_index(drop=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

Train_Generator = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.3,
                                    zoom_range=0.2,
                                    brightness_range=[0.2,0.9],
                                    rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.1)

Test_Generator = ImageDataGenerator(rescale=1./255)

from sklearn.model_selection import train_test_split
Train_Data,Test_Data = train_test_split(df,train_size=0.9,random_state=42,shuffle=True)

'''
d={'Fire':1, 'No_fire':0}
Test_Data['label']=Test_Data['label'].map(d)
Test_Data.head()
'''

Train_IMG_Set = Train_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="image",
                                                   y_col="label",
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="training")

Validation_IMG_Set = Train_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="image",
                                                   y_col="label",
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="validation")

Test_IMG_Set = Test_Generator.flow_from_dataframe(dataframe=Test_Data,
                                                 x_col="image",
                                                 y_col="label",
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)

Model_Two = tf.keras.models.Sequential([
  # inputs
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(256,)),
  # hiddens layers
  tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  # output layer
  tf.keras.layers.Dense(2,activation="softmax")
])
Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5,mode="min")
Model_Two.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
ANN_Model = Model_Two.fit(Train_IMG_Set, validation_data=Validation_IMG_Set, callbacks=Call_Back, epochs=1
                        )


#front end
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import glob
import cv2
from tensorflow.keras.preprocessing import image


root= tk.Tk()

#Make a Canvas (i.e, a screen for your project

canvas = tk.Canvas(root, width = 750, height = 650)
canvas.configure(bg='grey19')
canvas.pack()

# App title label

label1 = tk.Label(root, bg="black", fg="grey81", text=' Fire Detection System ', font=("Segoe Script", 25), borderwidth=3, relief="solid")
canvas.create_window(375, 75, window=label1)

#alarm toggle button

#button1 = tk.Button (root, text='Toggle Alarm', bg='green') # button to call the 'values' command above
#canvas.create_window(705, 25, window=button1)

#Open file explorer

def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("JPG", "*.jpg*"),("all files", "*.*")))
#Open a new window for user
    open_win(filename)

#running model
def run():
    img = image.load_img(filename,target_size=(256,256))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    Pred = Model_Two.predict(x)
    Pred = Pred.argmax(axis=-1)
    print(Pred)
   

    #displaying output
    if Pred ==1:
        print('no fire')
        result = tk.Label(new, bg="green", fg="black", text="No Fire Detected", font=("Segoe Script", 25), borderwidth=3, relief="solid").pack()
    elif Pred ==0:
        print('fire')
        result = tk.Label(new, bg="red", fg="black", text="Fire Detected", font=("Segoe Script", 25), borderwidth=3, relief="solid").pack()
    else:
        result = tk.Label(new, bg="black", fg="grey81", text="Please Use a different Image", font=("Segoe Script", 25), borderwidth=3, relief="solid").pack()
        print('use diff image') 


def open_win(image):
    global new
    new= Toplevel(canvas)
    new.geometry("1000x850")
    new.title("New Window")
    Label(new, text="Selected image ", font=('Helvetica 17 bold',40)).pack(pady=20)

    #create run button
    button2 = tk.Button (new, text='Run', font=("ROG FONTS", 10), bg='green', command = lambda:run())
    button2.pack(pady=15)

   #inserting user image:
    frame = Frame(new, width=400, height=400)
    frame.pack()
    frame.place(anchor='center', relx=0.5, rely=0.5)
    img = ImageTk.PhotoImage(Image.open(filename))
    label = tk.Label(new, image = img).pack()
    new.mainloop()



#upload image

button1 = tk.Button (root, text='Upload Image', font=("ROG FONTS", 10), bg='green', command = browseFiles)
canvas.create_window(375, 150, window=button1)

root.mainloop()


#insert fireimage
from PIL import ImageTk, Image
import os
cwd_join = os.getcwd() + "\\"
icon_rel = os.path.relpath('C:\\Users\\Haame\\Documents\\AIIP\\Fire.jpg')
icon_abs = cwd_join + icon_rel
p1 = ImageTk.PhotoImage(file=icon_abs + "\\Fire.jpg")


frame = Frame(canvas, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)
img = ImageTk.PhotoImage(Image.open("C://Users//Haame//Documents//AIIP//Fire.jpg"))
label = Label(frame, image = pl).pack()

root.mainloop()

