from crypt import methods
from flask import Blueprint, app, render_template, request, flash, jsonify, redirect, url_for
from flask_login import login_required, current_user
from src.model import Note, Image
from . import db
import json
import os
from werkzeug.utils import secure_filename
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import matplotlib.pyplot as plt
#import cv2


UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'gif', 'png', 'jpg', 'jpeg', 'tif'}

views = Blueprint('views', __name__)

line_list = []
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def thresholding(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    plt.imshow(thresh, cmap='gray')
    return thresh

def get_image(filename):
    img = cv2.imread(f'src/static/uploads/{filename}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    
    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    thresh_img = thresholding(img)
    kernel = np.ones((3,50), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    img2 = img.copy()
    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        line_list.append([x, y, x+w, y+h])
        if h > 20 and w > 200:
            cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        
    text_list = []
    for i in range(0 ,len(line_list)):
        line = line_list[i]
        roi = img[line[1]:line[3], line[0]:line[2]]
        plt.imshow(roi)
        cv2.imwrite(f'output{i}.png', roi)
        text = text_recognition(f'output{i}.png')
        text_list.append(text)
        
    generated_text = ' '.join[text_list]
    return generated_text



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_recognition(filename):
    from PIL import Image
    url = f'src/static/uploads/{filename}'
    image = Image.open(url).convert("RGB")
    #processor = TrOCRProcessor.from_pretrained("src/processor-2") 
    #model = VisionEncoderDecoderModel.from_pretrained("src/model")
    #processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
    #model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # model = torch.load('src/model-2.pt', map_location=torch.device('cpu'))
    #model.to(device)

    pixel_values = processor(image, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    return generated_text


# @views.route('/', methods=['GET', 'POST'])
# @login_required
# def home():
#     return render_template("home.html", user=current_user)

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})

@views.route("/deleteImage/<int:id>")
def deleteImage(id):
    item = Image.query.get_or_404(id)
    db.session.delete(item)
    db.session.commit()
    return redirect("/notes")

@views.route("/deleteNotes/<int:id>")
def deleteNote(id):
    item = Note.query.get_or_404(id)
    db.session.delete(item)
    db.session.commit()
    return redirect("/notes")

@views.route('/delete-image', methods=['POST'])
def delete_image():
    note = json.loads(request.files)
    imageId = note['imageId']
    image = Image.query.get(imageId)
    if image:
        if image.user_id == current_user.id:
            db.session.delete(image)
            db.session.commit()

    return jsonify({})


@views.route('/', methods=['GET', 'POST'])
def landing():
    return render_template('landing.html', user=current_user)

@views.route('/home', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        file.save(os.path.join("src/static/uploads", secure_filename(file.filename)))
        if file.filename == '':
            flash('No image selected for uploading', category='error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload = Image(filename=file.filename, user_id=current_user.id)
            #print('upload_image filename: ' + filename)
          #  flash('Image successfully uploaded and displayed below')
            print('definitely here')
            
            pred = text_recognition(filename)
           # pred = "text_recognition(filename)"
           # pred = get_image(filename)
            upload.label = pred
            db.session.add(upload)
            db.session.commit()
            return render_template('home.html', filename=filename,pred=pred, user=current_user)
        else:
            print("Not avaliable")
            flash('Allowed image types are - png, jpg, jpeg, gif', category='error')
            return redirect('/home')
    else:
      return render_template('home.html', user=current_user)

 
@views.route('src/static/uploads/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def predict(filename):
    pred = text_recognition(filename)
    

#@views.route('/download')
def download():
    path = 'samplefile.pdf'
    return send_file(path, as_attachment=True)


@views.route('/contribute')
def contribute():
    
    return render_template('contribute.html', user=current_user)
    
    