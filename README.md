# Visual Question Answering system

In our project on the VQA system, we worked with a [dataset](https://www.kaggle.com/datasets/bhavikardeshna/visual-question-answering-computer-vision-) of real-world images and explored 6 model combinations to determine the most effective approach. We experimented with two text encoders — BERT and LSTM — and three image encoders: a custom CNN, pretrained ResNet-50, and Vision Transformer (ViT). The models were integrated using a multi-head attention mechanism. Among these, the combination of BERT and ViT performed best, achieving an accuracy of 28%. While this result is moderate, it highlights the inherent difficulty of processing real images in VQA tasks, possibly due to their complexity and diversity compared to synthetic datasets. The project provided valuable insights into model performance trade-offs and underscored the potential of transformer-based architectures for both text and image encoding.


## How to use

1. Create environment and install requirements

```
python -m venv venv
venv\Scripts\activate   # for Windows

pip install -r requirements.txt
```

2. Install package dependencies

```
pip install -e .
```

2. Run the code

```
python src\model\main.py
```