■　e2dproject-djangoをGitHub上げる際の手順

（１）\apps\appml\sub\algoフォルダをalgo-DUMMYフォルダで置換
      
（２）templates/appml/area4Inquire_appml.htmlの統計ボタン（他）disabled

（７）settings.py編集

（８）logファイルクリア
（９）readme.txt削除

------------------------------------------------
azure

ALLOWED_HOSTS = ['e2dj.azurewebsites.net', 'localhost', '127.0.0.1']  # Azure
CSRF_TRUSTED_ORIGINS = ['https://e2dj.azurewebsites.net']  # Azure

------------------------------------------------
■　初期化手順
スーパーユーザーkate.walsh@example.com作成

python.exe manage.py runserver 8000

http://127.0.0.1:8000/adminログイン
sakura.suwa@example.com作成

エクセルデータ登録 by kate
エクセルデータ登録 by sakura
