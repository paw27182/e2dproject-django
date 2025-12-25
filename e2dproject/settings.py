from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-juwz19_zqqqf)*5xg#jh4v%&)eeg4_@0*_2n8l8y0zr20!_!!t'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'apps.topview.apps.TopviewConfig',
    'apps.appmain.apps.AppmainConfig',
    'accounts.apps.AccountsConfig',
    'apps.appml.apps.AppmlConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'e2dproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [Path(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'e2dproject.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'database/db_admin.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


STATICFILES_DIRS = (
    Path(BASE_DIR, "static"),
)

# LOGIN_REDIRECT_URL = 'topview'
# LOGOUT_REDIRECT_URL = 'topview'

# Logger:Handler=1:N, Handler:Formatter=1:1
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    # set log format
    'formatters': {
        'production': {
            'format': '%(asctime)s [%(levelname)s] %(process)d %(thread)d '
                      '%(pathname)s:%(lineno)d %(message)s'
        },
        'dev': {
            'format': '\t'.join([
                '%(asctime)s',
                '[%(levelname)s]',
                '%(pathname)s(Line:%(lineno)d)',
                '%(message)s'
            ])
        },
    },
    # set handlers
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': Path(BASE_DIR, "log", "app.log"),
            # 'formatter': 'production',
            'formatter': 'dev',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            # 'formatter': 'production',
            'formatter': 'dev',
        },
    },
    # set loggers
    'loggers': {
        # logger for private applications
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        # logger for django
        'django': {
            # 'handlers': ['file'],
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

AUTH_USER_MODEL = 'accounts.User'

# In case of Windows 10
PYTHON_EXE_FILE = r"C:/Python/env/Scripts/python.exe"  # specify python executable file
DB_ADMINISTRATOR = ["kate.walsh@example.com", "mack.davis@example.com"]
DATABASE_TYPE = "SQLite3"

# # In case of Ubuntu 20.04.6 LT
# PYTHON_EXE_FILE = "/home/paw/env/bin/python3.10"  # specify python executable file
# DB_ADMINISTRATOR = ["kate.walsh@example.com", "mack.davis@example.com"]
# DATABASE_TYPE = "SQLite3"

# # In case of Azure(Linux)
# DB_ADMINISTRATOR = ["kate.walsh@example.com", "mack.davis@example.com"]
# DATABASE_TYPE = "SQLite3"
