from flask import Flask, request, send_file
import pixellib
from pixellib.tune_bg import alter_bg

app = Flask(__name__)

bg_remover = alter_bg()
bg_remover.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    file = request.files["image"]
    input_path = "input.jpg"
    output_path = "output.png"

    file.save(input_path)
    bg_remover.color_bg(input_path, colors=(255, 255, 255), output_image_name=output_path)

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run()
