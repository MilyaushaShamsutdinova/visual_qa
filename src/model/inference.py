import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from src.model.model import VisualEncoder_ViT, BERTTextEncoder, VQA_Attention
from src.utils import *
import warnings
warnings.filterwarnings("ignore")


def initialize_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = get_number_of_classes()

    img_encoder = VisualEncoder_ViT(finetune=False).to(device)
    text_encoder = BERTTextEncoder(finetune=False).to(device)

    model = VQA_Attention(
        target_size=n_classes,
        image_encoder=img_encoder,
        text_encoder=text_encoder,
        device=device,
        hidden_size=1024,
        n_layers = 0,
        dropout_prob=0.5,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=True)
    return model


def inference(model, image_path, question):
    """
    Perform VQA inference given a question and an image.
    
    Args:
        question (str): The question to ask about the image.
        image_path (str): Path to the image.
    
    Returns:
        str: Predicted answer to the question.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_classes = get_idx_to_classes()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Preprocess the question
    encoding = tokenizer(
        question,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image, input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        predicted_idx = predicted.item()

    answer = idx_to_classes[predicted_idx]
    return answer


# exmaple

# question = "what is in front of the chair"
# image_path = "images\image384.png"

# answer = inference(image_path, question)

# print(f"Question: {question}")
# print(f"Answer: {answer}")



# model = initialize_model(r'src\weights\bert_vit_20epochs.pt')

# while True:
#     image = str(input("Image name: "))
#     image_path = f"images\{image}"
#     question = str(input("Question: "))

#     if question == "stop":
#         break

#     s = time.time()
#     answer = inference(model, image_path, question)
#     e = time.time()

#     print(f"Answer: {answer}")
#     print(f"--execution time: {e-s:.2f} sec--\n")
