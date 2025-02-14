import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from torch.utils.checkpoint import checkpoint
from src.utils import *
import warnings
warnings.filterwarnings("ignore")


class BERTTextEncoder(nn.Module):
    def __init__(self, finetune=True):
        """
        Initializes the BERT-based text encoder.
        
        Args:
            finetune (bool): Whether to fine-tune the BERT model.
        """
        super(BERTTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = finetune
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for text encoding.
        
        Args:
            input_ids (Tensor): Tokenized input IDs.
            attention_mask (Tensor): Attention mask for the input.

        Returns:
            Tensor: Encoded text features from BERT.
        """
        outputs = checkpoint(self.bert, input_ids, attention_mask, use_reentrant=False)
        return outputs.last_hidden_state[:, 0, :]


class VisualEncoder_ViT(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', finetune=False):
        """
        Initializes the Vision Transformer (ViT) based image encoder.
        
        Args:
            model_name (str): Name of the pre-trained ViT model to load.
            finetune (bool): Whether to fine-tune the ViT model.
        """
        super(VisualEncoder_ViT, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        for param in self.vit.parameters():
            param.requires_grad = finetune
        
    def forward(self, image):
        """
        Forward pass for image encoding.
        
        Args:
            image (Tensor): Input image tensor.

        Returns:
            Tensor: Encoded image features from ViT.
        """
        outputs = self.vit(image)
        return outputs.pooler_output


class MultimodalAttention(nn.Module):
    def __init__(self, image_feature_dim, question_feature_dim, hidden_dim):
        """
        Initializes the multimodal attention module.
        
        Args:
            image_feature_dim (int): Dimensionality of image features.
            question_feature_dim (int): Dimensionality of text features.
            hidden_dim (int): Hidden layer dimensionality for attention.
        """
        super(MultimodalAttention, self).__init__()
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.question_projection = nn.Linear(question_feature_dim, hidden_dim)
        self.attention_scores = nn.Linear(hidden_dim, 1)
        
    def forward(self, image_features, question_features):
        """
        Forward pass for multimodal attention.
        
        Args:
            image_features (Tensor): Encoded image features.
            question_features (Tensor): Encoded text features.

        Returns:
            Tuple[Tensor, Tensor]: Combined features and attention weights.
        """
        projected_images = self.image_projection(image_features)
        projected_questions = self.question_projection(question_features)
        attention_weights = self.attention_scores(
            torch.tanh(projected_images + projected_questions.unsqueeze(1))
        )
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_image_features = attention_weights * projected_images.unsqueeze(1)
        attended_image_features = attended_image_features.sum(dim=1)
        combined_features = attended_image_features + projected_questions
        return combined_features, attention_weights


class VQA_Attention(nn.Module):
    def __init__(self, target_size, image_encoder, text_encoder, device, hidden_size=1024,
                 n_layers=1, dropout_prob=0.2, img_shape=(1, 3, 224, 224), text_shape=(1, 512)):
        """
        Initializes the VQA model with attention.

        Args:
            target_size (int): Number of output classes (answers).
            image_encoder (nn.Module): Pre-trained image encoder.
            text_encoder (nn.Module): Pre-trained text encoder.
            device (torch.device): Device to run the model on.
            hidden_size (int): Hidden layer size for attention and classification.
            n_layers (int): Number of layers in the classification head.
            dropout_prob (float): Dropout probability.
            img_shape (Tuple[int]): Input image shape for calculating output shape.
            text_shape (Tuple[int]): Input text shape for calculating output shape.
        """
        super(VQA_Attention, self).__init__()
        self.image_encoder = image_encoder
        enc_img_out = get_output_shape(image_encoder, img_shape)[1]
        self.text_encoder = text_encoder
        enc_text_out = text_encoder(
            torch.randint(0, 1, text_shape).to(device),
            torch.randint(0, 1, text_shape).to(device)
        ).data.shape[1]
        
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = MultimodalAttention(enc_img_out, enc_text_out, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.lin2 = nn.Linear(hidden_size, target_size)
        
    def forward(self, image, text, mask=None):
        """
        Forward pass for the VQA model.
        
        Args:
            image (Tensor): Input image tensor.
            text (Tensor): Input text tensor.
            mask (Tensor): Attention mask for the text input.

        Returns:
            Tensor: Predicted answer logits.
        """
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text, mask)
        combined = self.attention(img_features, text_features)
        out, _ = combined
        for layer in self.hidden:
            out = self.dropout(out)
            out = layer(out)
            out = torch.nn.functional.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        return out
