# Flask-OpenCV-Registrasi-Face-Recognition
1. Create MySQL database:  First of all create a MySQL database with the name flask_db via phpMyAdmin. This PhpMyAdmin is part of XAMPP.Open phpMyAdmin via browser then Create database flask_db.

2. After successfully creating the flask_db database, copy the SQL script below to create a master personnel table prs_mstr and dataset image tables img_dataset. Then follow the steps as in the following picture:<br>![Click flask_db](https://github.com/md-irfan-hasan-fahim/Flask-OpenCV-Registrasi-Face-Recognition/assets/81842071/d4f771b9-367c-49a7-b510-68474b5efa1d)
 <br> and look like this: <br> ![table_created](https://github.com/md-irfan-hasan-fahim/Flask-OpenCV-Registrasi-Face-Recognition/assets/81842071/345a2fdd-c8cb-4210-a994-a7c0afb4786a)
 
3. Adds the accs_hist table in the flask_db database:  Think of this application as a room access control application, where everyone who will enter a restricted room must scan the face first. Data on personnel entering the restricted room will be recorded in the database. Here I add a new table with the name accs_hist (access history) as a table for storing incoming personnel data. Open phpMyAdmin, select the flask_db database, paste the SQL script into the query window in the SQL phpMyAdmin tab, then click the Go button. ![phpmyadmin_flaskdb](https://github.com/md-irfan-hasan-fahim/Flask-OpenCV-Registrasi-Face-Recognition/assets/81842071/33ac7448-a617-405e-8b31-4f98eea45711)
<br>
The accs_hist table was successfully created.<br>
![accs_hist_table](https://github.com/md-irfan-hasan-fahim/Flask-OpenCV-Registrasi-Face-Recognition/assets/81842071/596c29b6-b6ce-4380-8f69-1a6382b56c84)

4. Create Pycharm Project and Install Packages:   Create new project on any IDE then name the project FlaskOpenv_FaceRecognition. After that click the button Create

5.Package-packages that must be installed for this project face recognition include:<br>
  a.Flask<br>
  b.mysql-connector<br>
  c.opencv-python<br>
  d.opencv-contrib-python<br>
  e.Pillow<br>

