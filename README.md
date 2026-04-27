# Face-Attendance-System
AI-based Face Recognition Attendance System
Below is a pin-to-pin, exam-ready, copy-paste README.md that explains everything: packages, setup, commands, folder structure, and how to run the project successfully.

You can copy this exactly into a file named README.md in your project root and push it to GitHub.

📌 README.md — Face Attendance System
# 🎓 Face Attendance System

An AI-based **Face Recognition Attendance System** developed as a **Final Year Academic Project** using Python, Flask, OpenCV, and SQLite.  
The system supports face enrollment, live recognition, attendance tracking, admin management, reports, and a modern glassmorphic UI.

---

## 🚀 Features

### 👤 User Features
- Face Enrollment using webcam
- Live Face Recognition
- Automatic Attendance Marking
- Attendance History (Collapsible View)
- Download Attendance Report (PDF)
- Feedback Submission
- Grievance / Complaint Submission

### 🛠️ Admin Features
- Secure Admin Login
- View Attendance Analytics
- Automatic Payroll Management
- Monthly Salary Calculation from Attendance
- Download Full Attendance Report (PDF)
- Manage Complaints (Open / Close)
- View User Feedback
- Real-time Model Accuracy Monitoring
- Recognition Performance Analysis

### 🎨 UI & UX
- Modern Glassmorphic UI
- Responsive Design (Bootstrap 5)
- Collapsible Sections (Accordion)
- Charts & Analytics (Chart.js)
- Toast Notifications
- Clean & Exam-Ready Design

---

## 🧠 Technologies Used

- **Python 3.10+**
- **Flask** (Web Framework)
- **OpenCV (opencv-contrib-python)** (Face Recognition)
- **SQLite** (Database)
- **NumPy**
- **Pillow (PIL)**
- **ReportLab** (PDF generation)
- **Bootstrap 5**
- **Chart.js**
- **Font Awesome**
- **HTML / CSS / JavaScript**

---

## 📁 Project Folder Structure



FaceAttendance/
│
├── app.py
├── README.md
├── requirements.txt
│
├── database/
│ └── attendance.db
│
├── static/
│ ├── css/
│ │ └── theme.css
│ └── images/
│
├── templates/
│ ├── base.html
│ ├── index.html
│ ├── enroll.html
│ ├── attendance.html
│ ├── admin.html
│ ├── admin_complaints.html
│ ├── admin_feedback.html
│ ├── login.html
│ ├── feedback.html
│ ├── grievance.html
│ └── faq.html
│
├── TrainingImage/
├── TrainingImageLabel/
│ └── Trainner.yml
│
└── haarcascade_frontalface_default.xml


---

## ⚙️ Installation & Setup (Step-by-Step)

### ✅ Step 1: Install Python
Download and install Python from:
👉 https://www.python.org/downloads/

✔️ Make sure **“Add Python to PATH”** is checked during installation.

Verify:
```bash
python --version

✅ Step 2: Clone the Project (or Download ZIP)
git clone https://github.com/your-username/Face-Attendance-System.git
cd Face-Attendance-System


OR download ZIP and extract.

✅ Step 3: Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

✅ Step 4: Install Required Packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

📦 Required Python Packages

Create requirements.txt with this content:

flask
opencv-contrib-python
numpy
pillow
sqlite3
reportlab
python-dotenv

🗄️ Database Initialization (One Time)

Run this once to create required tables:

python init_db.py


You should see:

Database initialized successfully

📷 Haar Cascade File

Download Haar Cascade file from:
👉 https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

Save it in the project root folder:

FaceAttendance/haarcascade_frontalface_default.xml

▶️ How to Run the Project
python app.py


You should see:

Running on http://127.0.0.1:5000/


Open browser and visit:
👉 http://127.0.0.1:5000

🔐 Admin Login Credentials (Demo)
Username: admin
Password: admin123


⚠️ For academic/demo purposes only.

🧪 How the System Works

Enroll user face using webcam

System captures multiple face images

Model is trained automatically

Live recognition marks attendance

Attendance stored in SQLite database

Admin can view analytics & reports

📄 Reports & Exports

Attendance history downloadable as PDF

Payroll summaries downloadable as PDF

Admin can download full attendance report

Analytics shown using charts

🎓 Academic Use

This project is suitable for:

Final Year Project

Mini Project

AI / ML / Computer Vision Coursework

Flask Web Application Demonstration

👨‍💻 Author

Eswar
Final Year Student
AI-Based Face Attendance System

⭐ GitHub

If you find this project useful, please ⭐ the repository.


---

## ✅ FINAL STEPS (VERY IMPORTANT)

After creating `README.md`:

```bash
git add README.md
git commit -m "Add complete project README"
git push
