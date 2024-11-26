import torch
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANSWER_SPACE_PATH = r'src\data\answer_space.txt'


def get_output_shape(model, image_dim):
    """Get output shape of a model given an input image dimension."""
    return model(torch.rand(*(image_dim)).to(device)).data.shape


def get_classes_to_idx():
    """Load answer class-to-index mapping."""
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')
        return {cls_name: idx for idx, cls_name in enumerate(classes)}


def get_idx_to_classes():
    """Load index-to-answer class mapping."""
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')
        return {idx: cls_name for idx, cls_name in enumerate(classes)}


def get_number_of_classes():
    """Get the total number of answer classes."""
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')
        return len(classes)
