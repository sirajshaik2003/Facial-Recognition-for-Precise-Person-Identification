from flask import *
import mysql.connector
import pandas as pd
from PIL import Image
import cv2,os
import shutil
import datetime
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import time,os
from datetime import datetime

le = LabelEncoder()
app=Flask(__name__)
app.config['SECRET_KEY']='face recognition'
def data_base():
    db = mysql.connector.connect(host="localhost", user="root", passwd="", database="face_recognition")
    cur = db.cursor()
    return db,cur

facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        if username=="admin" and password=="admin":
            flash("Welcome Admin","success")
            return render_template('adminhome.html')
        else:
            flash("Invalid credentials entered","danger")
            return render_template('admin.html')
    return render_template('contact.html')

@app.route('/addperson',methods=["POST","GET"])
def addperson():
    if request.method=='POST':
        Id=request.form['empid']
        name=request.form['name']
        tel = request.form['tel']
        addr = request.form['addr']
        profile = request.files.get('profile')
        profile_path = os.path.join('static','profiles',f'{Id}.png')
       
        db,cur=data_base()
        if not Id:
           flash("Please enter roll number properly ","warning")
           return render_template('addemp.html')

        elif not name:
            flash("Please enter your name properly ", "warning")
            return render_template('addemp.html')
        elif not tel:
            flash("Please enter mobile number","warning")
        # elif (Id.isalpha() and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        path_to_store=os.path.join(os.getcwd(),"dataset\\"+Id)
        try:
            shutil.rmtree(path_to_store)
        except:
            pass

        os.makedirs(path_to_store, exist_ok=True)

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder 
                cv2.imwrite(path_to_store +r'\\'+ str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])        
            else:
                cv2.imshow('frame', img)
                # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                # break if the sample number is morethan 100
            elif sampleNum > 150:
                    break
        cam.release()
        cv2.destroyAllWindows()
        ts = time.time()
        profile.save(profile_path) # saving the image
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        sq="insert into person(EmpId,Name,Mobile,Address,Date,Time,image) values(%s,%s,%s,%s,%s,%s,%s)"
        val=(Id,name,tel,addr,date,timeStamp,f'/static/profiles/{Id}.png')
        cur.execute(sq,val)
        db.commit()
        flash("Data added successfully","success")
        return render_template("addperson.html")
    db,cur=data_base()
    s="select count(*) from person"
    ss=pd.read_sql_query(s,db)
    count=ss.values[0][0]
    if count==0:
        return render_template('addperson.html',empid='PID1')
    else:
        s1="select EmpId from person ORDER BY EmpId DESC LIMIT 1"
        ss1=pd.read_sql_query(s1,db)
        cid=str(ss1.values[0][0])
        text = ""
        numbers = ""
        digits = "0123456789"
        res = []
        for i in cid:
            if (i in digits):
                numbers += i
            else:
                text += i
        res.append(text)
        res.append(numbers)
        res=int(res[1])+1
        empid="PID"+str(res) 
    return render_template('addperson.html',empid=empid)

@app.route('/viewperson')
def viewperson():
    db,cur=data_base()
    sql="select * from person"
    cur.execute(sql)
    data=cur.fetchall()
    return render_template('viewperson.html',data=data)

@app.route('/deletion/<id>')
def deletion(id=""):
    db,cur=data_base()

    sql="delete from person where EmpId='"+id+"'"
    cur.execute(sql)
    db.commit()
    path_to_store=os.path.join(os.getcwd(),"dataset\\"+id)
    # folder_path = 'path_to_your_folder'
    try:
        shutil.rmtree(path_to_store)
        print(f"Folder '{path_to_store}' has been deleted successfully.")
    except:
        pass
    flash("data deleted","success")
    return redirect(url_for('viewperson'))

def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[1]
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
            # Ids.append(int(aadhar_id))
    Ids_new=le.fit_transform(Ids).tolist()
    output = open('./model/encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    return faces, Ids_new

def count_folders(directory):
    # List all entries in the given directory
    entries = os.listdir(directory)
    
    # Count entries that are directories
    folder_count = sum(os.path.isdir(os.path.join(directory, entry)) for entry in entries)
    
    return folder_count


@app.route('/train', methods=['POST','GET'])
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = getImagesAndLabels(r"dataset")
    # Specify the directory you want to check
    directory_to_check = 'dataset'

    # Count the folders and print the result
    folder_count = count_folders(directory_to_check)
    print(Id)
    print(len(Id))
    recognizer.train(faces, np.array(Id))
    recognizer.save("./model/Trained.yml")
    flash(r"Model Trained Successfully", 'Primary')
    return redirect(url_for('viewperson'))

@app.route('/recognition')
def recognition():
    pkl_file = open('./model/encoder.pkl', 'rb')
    my_le = pickle.load(pkl_file)
    pkl_file.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("./model/Trained.yml")
    cam=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    font = cv2.FONT_HERSHEY_SIMPLEX
    flag = 0
    detected_persons=[]
    global det
    det=0
    while True:
        ret, im = cam.read()
        flag += 1
        if flag==200:
            flash(r"Unable to detect person. Contact help desk for manual voting", "info")
            cv2.destroyAllWindows()
            return render_template('index.html')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            print(conf)
            if (conf < 40):
                det +=1
                empid = my_le.inverse_transform([Id])[0]
                detected_persons.append(empid)
                db,cur=data_base()
                sql="select Name,Address,Mobile from person where EmpId='"+str(empid)+"'"
                cur.execute(sql)
                data=cur.fetchall()
                name=data[0][0]
                address = data[0][1]
                mobile = data[0][2]
                db.commit()
                cur.close()
                if det==20:
                    ts = time.time()
                    timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    local = datetime.now()
                    date1 = datetime.now().strftime('%d-%m-%Y')
                    m1 = local.strftime("%B")
                    db,cur=data_base()
                    s = "select count(1) from face_recognition where emp_id='%s' and date1='%s'" % (empid, date1)
                    z = pd.read_sql_query(s, db)
                    count = z.values[0][0]
                    db.commit()
                    cur.close()
                    if count==0:
                        db,cur=data_base()
                        sq="insert into face_recognition(emp_id,name,date1,month,in_time,address,phone,image) values(%s,%s,%s,%s,%s,%s,%s,%s)"
                        val=(str(empid),str(name),str(date1),str(m1),str(timeStamp),address,mobile,f'/static/profiles/{empid}.png')
                        cur.execute(sq,val)
                        db.commit()
                        cur.close()
                        det=0
                        cam.release()
                        cv2.destroyAllWindows()
                        flash('Face Recognised successfully',"success")
                        return redirect(url_for('attendence' ,empid=empid))
                    else:
                        db,cur=data_base()
                        ss="select in_time from face_recognition where emp_id='%s' and date1='%s' "%(empid,date1)
                        cur.execute(ss)
                        data1=cur.fetchall()
                        in_time=str(data1[0][0])
                        db.commit()
                        cur.close()
                        intime = datetime.strptime(in_time, '%H:%M:%S')
                        outtime = datetime.strptime(timeStamp, '%H:%M:%S')

                        # Calculate the time difference
                        elapsed_time = outtime - intime
                        db,cur=data_base()
                        sq="update face_recognition set out_time='%s',overall_time='%s' where emp_id='%s' and date1='%s'"%(timeStamp,elapsed_time,empid,date1)
                        cur.execute(sq)
                        db.commit()
                        cur.close()
                        det=0
                        cam.release()
                        cv2.destroyAllWindows()
                        flash('Face Recognised successfully',"success")
                        return redirect(url_for('attendence' ,empid=empid))

            else:
                name="Unknown"
            cv2.putText(im, name, (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('im', im)
        try:
            cv2.setWindowProperty('im', cv2.WND_PROP_TOPMOST, 1)
        except:
            pass
        if (cv2.waitKey(1) == (ord('q')) ):
            break
    cam.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

@app.route("/attendence")
def attendence():
    empid = request.args.get('empid')
    sql = f"select emp_id,name,image,address,phone,in_time,out_time,overall_time,date1 from face_recognition where emp_id='{empid}'"
    _,cursor = data_base()
    cursor.execute(sql)
    data = cursor.fetchall()
    cursor.close()
    return render_template('attendence.html',data=data)

@app.route('/personlogins')
def personlogins():
    db,cur=data_base()
    sql="select * from face_recognition"
    cur.execute(sql)
    data=cur.fetchall()
    return render_template('personlogins.html',data=data)

if __name__=='__main__':
    app.run(debug=True)