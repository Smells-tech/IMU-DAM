from flask import Flask, request, send_from_directory
from demo import generate as generate_fake, load_checkpoints, DefaultOptions
import os
from multiprocessing import Process

app = Flask(__name__)
app.config["RESULT_VIDEOS"] = "/home/luuk/development/first-order-model/results"

os.makedirs(os.path.join(app.instance_path, 'upload'), exist_ok=True)
opt=DefaultOptions()
generator, kp_detector = load_checkpoints(
        config_path=opt.config,
        checkpoint_path=opt.checkpoint,
        cpu=opt.cpu
    )

@app.route('/')
def hello_world():
    return 'done!'


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if  request.files:
            file = request.files["image"]
            file.save(os.path.join(app.instance_path, 'upload', "source.png"))
            return "OK", 200
    else:
        return "NOT OK", 500


@app.route("/generate/<index>")
def generate(index):
    generate_fake(generator, kp_detector, driver_index=int(index))
    return send_from_directory(app.config['RESULT_VIDEOS'], 'result-'+index+'.mp4')

if __name__ == '__main__':
    app.run(host="0.0.0.0")