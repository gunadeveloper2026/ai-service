import os
from flask import Flask, request, jsonify
from recognition import recognize_face

app = Flask(__name__)

@app.get('/')
def root():
    return jsonify({'status': 'ok', 'service': 'face-ai-service'})

@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/recognize', methods=['POST'])
def recognize():
    # expect image file in form-data
    file = request.files.get('image')
    if not file:
        return jsonify({'error':'no image'}), 400
    # optional gallery JSON in form field 'gallery'
    gallery = request.form.get('gallery')
    try:
        result = recognize_face(file.stream, gallery=gallery)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port, debug=True)
