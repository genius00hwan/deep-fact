import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from evaluate import return_acc

app = Flask(__name__, static_folder='static', static_url_path='/images')

# 업로드 디렉토리 설정
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# 디렉토리가 없는 경우 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/fileUpload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # 'file' 키로 전송된 파일 가져오기
        if 'file' not in request.files:
            return 'No file part in the request'

        file = request.files['file']

        # 파일이 선택되지 않았거나 빈 파일 처리
        if file.filename == '':
            return 'No selected file'

        # 파일 확장자 확인
        if not allowed_file(file.filename):
            return 'Unsupported file type'

        # 안전한 파일명으로 저장
        custom_filename = f"upload_file_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], custom_filename)

        # 업로드 폴더가 없으면 생성
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file.save(filepath)

        # 파일 경로를 return_acc 함수로 전달
        output = return_acc(filepath)

        # 결과 반환 (예: 'yes' 또는 'no')
        return make_result(output)

    return hello_world()


def make_result(output):
    return render_template('result.html', result='yes' if output == 1 else 'no')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
