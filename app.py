from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from pymongo import MongoClient
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date, datetime
import sys

app = Flask(__name__)
 
cnt = 0
pause_cnt = 0
justscanned = False
 
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["flask_db"]
    print("Connected to MongoDB successfully!")
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    sys.exit(1)
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("C:/Users/Erik/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    last_img = db.img_dataset.find_one(sort=[("img_id", -1)])
    lastid = last_img['img_id'] if last_img else 0
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            try:
                db.img_dataset.insert_one({
                    "img_id": img_id,
                    "img_person": nbr
                })
            except Exception as e:
                print(f"Error inserting data into MongoDB: {e}")
                # Handle the error appropriately, e.g., logging or user notification
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/Erik/PycharmProjects/FlaskOpencv_FaceRecognition/dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
 
    return redirect('/')
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        global justscanned
        global pause_cnt
 
        pause_cnt += 1
 
        coords = []
 
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
 
            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1
 
                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w
 
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
 
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
 
                try:
                    result = db.img_dataset.aggregate([
                        {"$match": {"img_id": id}},
                        {"$lookup": {
                            "from": "prs_mstr",
                            "localField": "img_person",
                            "foreignField": "prs_nbr",
                            "as": "person_info"
                        }},
                        {"$unwind": "$person_info"},
                        {"$project": {
                            "img_person": 1,
                            "prs_name": "$person_info.prs_name",
                            "prs_skill": "$person_info.prs_skill"
                        }}
                    ])
                    row = next(result, None)
                    if row:
                        pnbr = row['img_person']
                        pname = row['prs_name']
                        pskill = row['prs_skill']
                    else:
                        print(f"No data found for id: {id}")
                        pnbr = pname = pskill = None
                except Exception as e:
                    print(f"Error querying database: {e}")
                    pnbr = pname = pskill = None
 
                if int(cnt) == 30:
                    cnt = 0
 
                    try:
                        db.accs_hist.insert_one({
                            "accs_date": str(date.today()),
                            "accs_prsn": pnbr
                        })
                    except Exception as e:
                        print(f"Error inserting access history: {e}")
                        # Consider adding appropriate error handling or logging here
 
                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)
 
                    justscanned = True
                    pause_cnt = 0
 
            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
 
                if pause_cnt > 80:
                    justscanned = False
 
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("C:/Users/Erik/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
 
    wCam, hCam = 400, 400
 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)
 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
        key = cv2.waitKey(1)
        if key == 27:
            break
 
 
 
@app.route('/')
def home():
    try:
        data = list(db.prs_mstr.find({}, {'_id': 0, 'prs_nbr': 1, 'prs_name': 1, 'prs_skill': 1, 'prs_active': 1, 'prs_added': 1}))
    except Exception as e:
        print(f"Error fetching data: {e}")
        data = []
 
    return render_template('index.html', data=data)
 
@app.route('/addprsn')
def addprsn():
    try:
        result = db.prs_mstr.find().sort("prs_nbr", -1).limit(1)
        max_nbr = result[0]['prs_nbr'] if result.count() > 0 else 100
        nbr = max_nbr + 1
    except Exception as e:
        print(f"Error fetching max prs_nbr: {e}")
        nbr = 101  # Default value if there's an error
    # print(int(nbr))
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    try:
        prs_mstr_collection = db['prs_mstr']
        new_person = {
            'prs_nbr': prsnbr,
            'prs_name': prsname,
            'prs_skill': prsskill
        }
        result = prs_mstr_collection.insert_one(new_person)
        if not result.inserted_id:
            raise Exception("Failed to insert new person")
    except Exception as e:
        print(f"Error inserting new person: {e}")
        return jsonify({'error': 'Failed to add new person'}), 500
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    try:
        pipeline = [
            {
                "$match": {
                    "accs_date": datetime.now().strftime("%Y-%m-%d")
                }
            },
            {
                "$lookup": {
                    "from": "prs_mstr",
                    "localField": "accs_prsn",
                    "foreignField": "prs_nbr",
                    "as": "person_info"
                }
            },
            {
                "$unwind": "$person_info"
            },
            {
                "$project": {
                    "accs_id": 1,
                    "accs_prsn": 1,
                    "prs_name": "$person_info.prs_name",
                    "prs_skill": "$person_info.prs_skill",
                    "accs_added": 1
                }
            },
            {
                "$sort": {"accs_id": -1}
            }
        ]
        data = list(mongo.db.accs_hist.aggregate(pipeline))
    except Exception as e:
        app.logger.error(f"Error fetching data: {str(e)}")
        data = []
 
    return render_template('fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def countTodayScan():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['flask_db']
        collection = db['accs_hist']

        today = datetime.now().date()
        rowcount = collection.count_documents({'accs_date': today})

        return jsonify({'rowcount': rowcount})
    except Exception as e:
        app.logger.error(f"Error in countTodayScan: {str(e)}")
        return jsonify({'error': 'An error occurred while counting today\'s scans'}), 500
    finally:
        if 'client' in locals():
            client.close()
 
 
@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["flask_db"]
        accs_hist = db["accs_hist"]
        prs_mstr = db["prs_mstr"]

        pipeline = [
            {
                "$match": {
                    "accs_date": datetime.now().strftime("%Y-%m-%d")
                }
            },
            {
                "$lookup": {
                    "from": "prs_mstr",
                    "localField": "accs_prsn",
                    "foreignField": "prs_nbr",
                    "as": "person_info"
                }
            },
            {
                "$unwind": "$person_info"
            },
            {
                "$project": {
                    "accs_id": 1,
                    "accs_prsn": 1,
                    "prs_name": "$person_info.prs_name",
                    "prs_skill": "$person_info.prs_skill",
                    "accs_added": {
                        "$dateToString": {
                            "format": "%H:%M:%S",
                            "date": "$accs_added"
                        }
                    }
                }
            },
            {
                "$sort": {"accs_id": -1}
            }
        ]

        data = list(accs_hist.aggregate(pipeline))
        return jsonify(response=data)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if client:
            client.close()
 
    return jsonify(response = data)
 
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
