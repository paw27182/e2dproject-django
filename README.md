# Excel to Database application

# 1. What you can do

* To accumulate excel forms to the database.<br>
* To learn the django application.

<br>

<img src="e2d.png">

<br>

# 2. How to use

* Install the prerequisite python libraries

* Environment setting
  * Edit settings.py
    * set PYTHON_EXE_FILE

<br>

* Program start
  * cd e2dproject-django
  * python.exe manage.py runserver 8000

<br>

* Open browser
  * http://localhost:8000/
  * Login Email address/Password: sakura.suwa@example.com/helloe2d

<br>

# 3. System
* OS: Windows 10, Ubuntu 20.04.6 LTS)
* Web Framework: Django
* Python 3.12.7
* Python Libraries: See requirements.txt
* Bootstrap 5.2.3
* jQuery 3.7.1
* Database: SQLite3

<br>

# 4. Undisclosed Functions
* Authentication(Sign up, Change password)
* Database(PostgreSQL, MongoDB)
* Machine Learning

<br>

# 5. Initialization
(1) To sign-up a database administrator [UNDISCLOSED]<br>
(2) The database administrator submits the following control files: [UNDISCLOSED]<br>
   * database_info_entry.xlsx<br>
   * group_info_entry.xlsx(group administrators information)<br>

(3) To sigh-up a group administrator [UNDISCLOSED]<br>
(4) The group administrator submits the following control files:<br>
  * form_info_entry.xlsx(sumittable forms information）<br>
  * group_info_entry.xlsx(user information)

<br>

# 6. Directories and Files Overview

| Directory/File |D/F| description |
| :------------- | :-| :---------- |
| accounts |Dir | login processing |
| apps/appmain | Dir | main programs |
| apps/appml | Dir | machine learning [UNDISCLOSED] |
| apps/topview | Dir ||
| database | Dir ||
| download | Dir | sample data files |
| e2dproject | Dir | Django settings.py, etc. |
| log | Dir ||
| static | Dir ||
| templates | Dir ||
| manage.py  | File ||
| READMD.md | File ||
| requirements.txt | File | prerequisite libraries |
