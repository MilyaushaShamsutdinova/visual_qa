import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from src.model.model import VisualEncoder_ViT, BERTTextEncoder, VQA_Attention
from src.utils import *
import warnings
warnings.filterwarnings("ignore")


def initialize_model(model_path):
    """
    Initializes the VQA model based on Visual Transformer and BERT with pre-trained weights.

    Args:
        model_path (str): Path to the pre-trained model weights.

    Returns:
        torch.nn.Module: Loaded VQA model ready for inference.
    """
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
        n_layers=0,
        dropout_prob=0.5,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=True)
    model.eval()
    return model


def inference(model, image_path, question):
    """
    Perform VQA inference given a question and an image.

    Args:
        model (torch.nn.Module): The pre-trained VQA model.
        image_path (str): Path to the input image.
        question (str): The question to ask about the image.

    Returns:
        str: Predicted answer to the question.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_classes = get_idx_to_classes()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Preprocess the question - tokenize and create attention masks
    encoding = tokenizer(
        question,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Preprocess the image - resize, normalize, and convert to tensor
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
