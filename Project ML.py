import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
##aa
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet




#####################################################################################################

def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("jpg", "*.jpg*"), ("png", "*.png")))
    basewidth = 300 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)#image preprocessing remove noise
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper(), padx=20, pady=5,).pack()
    panel_image = tk.Label(frame, image=img).pack()


#####################################################################################################

def classify():
    from keras.applications.resnet import ResNet50
    from keras.applications.resnet import preprocess_input
    from keras.applications.resnet import decode_predictions


    model = ResNet50()
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)#image antialas reduse noise in image
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)##يطول حجم المصفوفه و المحور صفر هو المحور الاول
    processed_image =preprocess_input(image_batch.copy())

    original = preprocess_input(processed_image)
    yhat = model.predict(original)
    label = decode_predictions(yhat)
    
    label = decode_predictions(yhat)
    table = tk.Label(frame, text="Top image class predictions of ResNet_50").pack()

    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' + 
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()
    table = tk.Label(frame,bg="yellow", text="***********************************************").pack()
    


#####################################################################################################
def classify2():
    

    model = VGG16()
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)#يطول حجم المصفوفه و المحور صفر هو المحور الاول
    processed_image =preprocess_input(image_batch.copy())

    original = preprocess_input(processed_image)
    yhat = model.predict(original)
    label = decode_predictions(yhat)
    
    label = decode_predictions(yhat)
    table = tk.Label(frame, text="Top image class predictions of VGG_16").pack()
    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' + 
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()
    

    table = tk.Label(frame,bg="yellow", text="***********************************************").pack()
 
   

#####################################################################################################
def classify3():
    
    model = MobileNet()
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)#image antialas reduse noise in image
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)#يطول حجم المصفوفه و المحور صفر هو المحور الاول
    processed_image =preprocess_input(image_batch.copy())

    original = preprocess_input(processed_image)
    yhat = model.predict(original)
    label = decode_predictions(yhat)
    
    label = decode_predictions(yhat)
    table = tk.Label(frame, text="Top image class predictions of MobileNet").pack()
    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' + 
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()
    

    table = tk.Label(frame,bg="yellow", text="***********************************************").pack()
#####################################################################################################
def classify4():
    from keras.applications.densenet import DenseNet121
    from keras.applications.densenet import preprocess_input
    from keras.applications.densenet import decode_predictions


    model =DenseNet121()
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image =preprocess_input(image_batch.copy())

    original = preprocess_input(processed_image)
    yhat = model.predict(original)
    label = decode_predictions(yhat)
    
    label = decode_predictions(yhat)
    table = tk.Label(frame, text="Top image class predictions of DeseNet121").pack()
    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' + 
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()
 
    table = tk.Label(frame,bg="yellow", text="***********************************************").pack()

#####################################################################################################
def classify5( ):
    from keras.applications.inception_resnet_v2 import preprocess_input
    from keras.applications.inception_resnet_v2 import decode_predictions


    model = InceptionResNetV2()
    
    original = Image.open(image_data)
    original = original.resize((299,299))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image =preprocess_input(image_batch.copy())

    original = preprocess_input(processed_image)
    yhat = model.predict(original)
    label = decode_predictions(yhat)
    
    label = decode_predictions(yhat)
    table = tk.Label(frame, text="Top image class predictions of InceptionResNetV2").pack()
    for i in range(0, len(label[0])):
         result = tk.Label(frame,
                    text= str(label[0][i][1]).upper() + ': ' + 
                           str(round(float(label[0][i][2])*100, 3)) + '%').pack()
    table = tk.Label(frame,bg="yellow", text="***********************************************").pack()

       

#####################################################################################################
# gui
#
root = tk.Tk()
root.title(' Image Classifier with ML')
root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Image Classifier with ML", padx=20, pady=5, font=("bold", 25)).pack()
canvas = tk.Canvas(root, height=650, width=1500, bg='green')
canvas.pack()
frame = tk.Frame(root, bg='maroon')
frame.place(relwidth=.8, relheight=.8, relx=0.1, rely=0.1)
#####################################################################################################

chose_image = tk.Button(root, text='Choose Image',
                        padx=50, pady=15,
                        fg="white", bg="red",font=("bold", 17) ,command=load_img)
chose_image.pack(side=tk.LEFT)
#####################################################################################################
class_image = tk.Button(root, text='VGG_16',
                        padx=50, pady=15,
                        fg="white", bg="blue",font=("bold", 15) ,command=classify2)
class_image.pack(side=tk.RIGHT)
#####################################################################################################
class_image = tk.Button(root, text='ResNet_50',
                        padx=50, pady=15,
                        fg="white", bg="blue", font=("bold", 15),command=classify)
    

class_image.pack(side=tk.RIGHT)
#####################################################################################################
class_image = tk.Button(root, text='MobilNet',
                        padx=50, pady=15,
                        fg="white", bg="blue",font=("bold", 15) ,command=classify3)
class_image.pack(side=tk.RIGHT)
#####################################################################################################


class_image = tk.Button(root, text='DeseNet121',
                        padx=50, pady=15,
                        fg="white", bg="blue", font=("bold", 15),command=classify4)
class_image.pack(side=tk.RIGHT)


#####################################################################################################
class_image = tk.Button(root, text='InceptionResNetV2',
                        padx=50, pady=15,
                        fg="white", bg="blue", font=("bold", 15),command=classify5)
class_image.pack(side=tk.RIGHT)
#####################################################################################################
root.mainloop()

