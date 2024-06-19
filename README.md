# Valley Fever Dust Density Prediction

## Overview
This project leverages a Convolutional Neural Network to precisely predict dust storm occurrences, focusing on reducing the health impacts associated with Valley Fever in the Southwest United States. Developed using Python and TensorFlow, it maintains an accuracy rate of 92.3%. The model processes images from NASA's EOSDIS satellite data, utilizing Keras to construct the CNN architecture and NumPy to manipulate data.
## Model
Utilizes a custom implementation of the AlexNet architecture trained on a labeled dataset of satellite images categorized into 'High Dust' and 'Low Dust'.

## Installation
pip install -r requirements.txt

## Usage
- Train the model using `model.py`.
- Predict dust density in new images with `main.py`.

## Dataset
The dataset is divided into training and testing sets, located in 'train' and 'test' directories, respectively.

## Contributing
Contributions are welcome. Please submit a pull request or open an issue for suggestions or bug reports.

## License
[MIT License](LICENSE)

