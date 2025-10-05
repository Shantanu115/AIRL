
## Repository Structure
```
├── q1.ipynb                     
├── q2.ipynb                       
└── README.md                    
```

## Q1: Vision Transformer on CIFAR-10 (PyTorch)

### Overview
Implementation of Vision Transformer (ViT) for CIFAR-10 image classification achieving competitive performance through custom architecture optimization and advanced training techniques.

### How to Run in Google Colab
1. Open `q1.ipynb` in Google Colab
2. Ensure GPU runtime is enabled: **Runtime → Change runtime type → GPU**
3. Run all cells sequentially from top to bottom
4. Training will complete in approximately 100 epochs

### Best Model Configuration

# Optimized hyperparameters
image_size = 32          # CIFAR-10 image dimensions
patch_size = 4           # Creates 8×8 grid of patches  
num_classes = 10         # CIFAR-10 classes
embed_dim = 384          # Embedding dimension
num_heads = 8            # Multi-head attention heads
num_layers = 8           # Transformer encoder layers
mlp_ratio = 4            # MLP expansion ratio
dropout_rate = 0.1       # Dropout for regularization
batch_size = 128         # Training batch size
epochs = 100             # Training epochs
learning_rate = 3e-4     # AdamW learning rate
weight_decay = 0.05      # L2 regularization
```

### Architecture Details
- **Patch Embedding**: Divides 32×32 images into 4×4 patches (64 total patches)
- **Positional Encoding**: Learnable positional embeddings added to patch embeddings
- **Transformer Blocks**: 8 encoder layers with multi-head self-attention (8 heads each)
- **Classification Head**: Global average pooling of patch representations → Linear layer
- **Regularization**: Dropout (0.1), weight decay (0.05), and label smoothing (0.1)

### Data Augmentation Pipeline
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2471, 0.2435, 0.2616])
])
```

### Training Configuration
- **Optimizer**: AdamW with learning rate 3e-4 and weight decay 0.05
- **Scheduler**: MultiStepLR with decay at epochs 50 and 75 (γ=0.1)
- **Loss Function**: CrossEntropyLoss with label smoothing (α=0.1)
- **Device**: CUDA GPU for training acceleration

### Results Summary
| Metric           | Final Value        |
|------------------|--------------------|
| **Test Accuracy**| **79.55%**         |
| Train Accuracy   | 99.47%             |
| Final Train Loss | 0.5274             |
| Training Time    | ~100 epochs on GPU |

### Key Implementation Features
- **Custom ViT Architecture**: Optimized for CIFAR-10's small image size
- **Advanced Regularization**: Dropout, weight decay, and label smoothing
- **Efficient Training**: Mixed precision and optimized data loading
- **Comprehensive Logging**: Training progress tracking and visualization

## Performance Summary
The Vision Transformer implementation achieves **79.55% test accuracy** on CIFAR-10, demonstrating effective adaptation of transformer architecture for small-scale image classification with appropriate regularization strategies and architectural modifications.

## Q2: Text-Driven Image Segmentation with CLIPSeg


### How to Run in Google Colab
1. Open `q2.ipynb` in Google Colab
2. Run the installation cell for required dependencies
3. Execute all cells sequentially
4. Modify image URLs and text prompts as desired

### Technical Implementation

#### Core Pipeline
# Model initialization
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Segmentation pipeline
def segment_image(image, text_prompt):
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probability mask
    mask = outputs.logits.squeeze().sigmoid().cpu().numpy()
    return mask
```

#### Pipeline Components
1. **Image Loading**: Supports both URL and local file loading with error handling
2. **Text Processing**: Natural language prompts processed through CLIP text encoder
3. **Segmentation**: CLIPSeg model generates pixel-wise segmentation masks
4. **Visualization**: Automatic mask overlay with customizable colors and transparency

### Features and Capabilities
- **Zero-shot Segmentation**: No training required for new object categories
- **Natural Language Interface**: Intuitive text-based object specification
- **Flexible Input**: URL-based and local image support
- **Visual Feedback**: Automatic overlay generation for result interpretation
- **Error Handling**: Robust image loading with fallback mechanisms


### Model Architecture
- **Base Model**: CIDAS/clipseg-rd64-refined (Hugging Face)
- **Text Encoder**: CLIP-based natural language understanding
- **Vision Encoder**: Modified ResNet backbone for image feature extraction
- **Decoder**: Lightweight segmentation head for mask generation

### Limitations and Considerations
- **Single Object Focus**: Optimized for single object segmentation per prompt
- **Prompt Sensitivity**: Performance varies with prompt specificity and clarity
- **Computational Requirements**: Requires sufficient GPU memory for optimal performance
- **Domain Constraints**: Limited by CLIPSeg's pre-training data distribution



