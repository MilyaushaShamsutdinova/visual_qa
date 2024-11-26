from src.model.inference import *
import time
import os
import warnings
warnings.filterwarnings("ignore")


model = initialize_model(r'src\weights\bert_vit_20epochs.pt')

while True:
    image = str(input("Image name: "))
    image_path = f"images\{image}"

    if not os.path.exists(image_path):
        print(f"Enter the valid image name!\n")
        continue

    question = str(input("Question: "))
    if question == "stop":
        break

    s = time.time()
    answer = inference(model, image_path, question)
    e = time.time()

    print(f"Answer: {answer}")
    print(f"--execution time: {e-s:.2f} sec--\n")
