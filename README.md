# Pretrained Image Classifier

Classifies pet images using pretrained CNN models (ResNet, AlexNet, VGG), compares predictions to the true labels from filenames, and reports how well each architecture performs on **dog vs. not-dog** and **dog breed** classification.

## Objectives

1. **Identify dogs vs. not-dogs** – For each image, determine whether the classifier correctly labels it as a dog or not-a-dog.
2. **Classify dog breeds** – For images that are dogs, measure how often the classifier predicts the correct breed.

True labels are derived from image filenames (e.g. `Boston_terrier_02303.jpg` → `boston terrier`). The program uses a dog-name list (`dognames.txt`) to decide which classifier outputs count as “dog.”

## Requirements

- Python 3
- PyTorch and torchvision (pretrained ResNet18, AlexNet, VGG16)
- Pillow (PIL)

All code and data live in the `data/` directory. Run commands from `data/` or use the paths below.

## Project Structure

```
pretrained_image_classifier/
├── README.md
└── data/
    ├── check_images.py          # Main program
    ├── classifier.py            # CNN inference (ResNet/AlexNet/VGG)
    ├── get_input_args.py        # Command-line arguments
    ├── get_pet_labels.py        # Build labels from filenames
    ├── classify_images.py       # Run classifier, compare labels
    ├── adjust_results4_isadog.py # Dog vs. not-dog flags
    ├── calculates_results_stats.py
    ├── print_results.py         # Summary and misclassification output
    ├── dognames.txt             # List of dog names/breeds
    ├── pet_images/              # 40 pet images (30 dogs, 10 non-dogs)
    ├── uploaded_images/         # Optional: your own images
    ├── run_models_batch.sh      # Run all 3 models on pet_images/
    ├── run_models_batch_uploaded.sh  # Run all 3 models on uploaded_images/
    └── PROJECT_RESULTS.md       # Detailed results and discussion
```

## Usage

**Single run (one architecture):**

```bash
cd data
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

**Batch run – all three models on `pet_images/` (output piped to text files):**

```bash
sh data/run_models_batch.sh
```

Produces: `data/resnet_pet-images.txt`, `data/alexnet_pet-images.txt`, `data/vgg_pet-images.txt`.

**Batch run – all three models on `uploaded_images/`:**

```bash
sh data/run_models_batch_uploaded.sh
```

Produces: `data/resnet_uploaded-images.txt`, `data/alexnet_uploaded-images.txt`, `data/vgg_uploaded-images.txt`.

## Results

Results below are from running `run_models_batch.sh` on the 40 images in `pet_images/` (30 dogs, 10 non-dogs).

| Statistic | ResNet | AlexNet | VGG |
|-----------|--------|---------|-----|
| N Images | 40 | 40 | 40 |
| N Dog Images | 30 | 30 | 30 |
| N NotDog Images | 10 | 10 | 10 |
| % Correctly Classified Dogs | 100.0 | 100.0 | 100.0 |
| % Correctly Classified Not-a-Dog | 90.0 | 100.0 | 100.0 |
| % Correctly Classified Dog Breeds | 90.0 | 80.0 | **93.3** |

- **Objective 1 (dog vs. not-dog):** VGG and AlexNet get 100% on both dogs and not-dogs. ResNet gets 100% on dogs but 90% on not-dogs (one misclassified non-dog).
- **Objective 2 (breed):** VGG is best at 93.3%; ResNet 90%; AlexNet 80%.

**Best model: VGG** – 100% on dog/not-dog and the highest breed accuracy (93.3%). Full discussion, objective-by-objective breakdown, and output file descriptions are in [data/PROJECT_RESULTS.md](data/PROJECT_RESULTS.md).
