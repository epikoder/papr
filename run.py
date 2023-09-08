import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from src import create_app
from src.views import UPLOAD_FOLDER


app = create_app()

if __name__ == '__main__':
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    

    
    
    app.run(debug=True, host='0.0.0.0')





