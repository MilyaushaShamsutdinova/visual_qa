import torch
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANSWER_SPACE_PATH = r'src\data\answer_space.txt'

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim)).to(device)).data.shape


def get_classes_to_idx():
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')

        classes_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(classes)
        }
        return classes_to_idx


def get_idx_to_classes():
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')

        idx_to_classes = {
            idx: cls_name for idx, cls_name in enumerate(classes)
        }
        return idx_to_classes


def get_number_of_classes():
    with open(ANSWER_SPACE_PATH, 'r') as f:
        data = f.read()
        classes = data.split('\n')
        return len(classes)
    