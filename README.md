# Fire Segmentation
[![website](https://img.shields.io/badge/whereisfire.pp.ua-website-green)](http://whereisfire.pp.ua)

### **1. GENERAL INFO**
<img src="pictures/main_updated.jpg" width="800"/>

Neural networks course project. Website to segment fire from image 
in case it contains it. Uses `resnet` for classification 
and `unet` for segmentation. The main idea is to get information
whether given image pictures fire and in case the answer is
positive - get image with segment highlighted.

<img src="pictures/plan.jpg" width="800"/>

For demonstration of our models we created a simple web
application using `flask`. User can load image from their 
file system and observe the result - message or mask

<img src="pictures/website.jpg" width="800"/>

### **2. DATASETS**
**2.1 ResNet dataset** \
That is a mix of three datasets:
- [FIRE Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset) - folders with fire images and non-fire images.
- [Fire Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/test-dataset) - folders with fire images and non-fire images.
- [BoWFire](https://bitbucket.org/gbdi/bowfire-dataset/downloads/) - folder with fire and non-fire images, and folder with fire masks for segmentation training. ResNet doesn't need any masks, so we don't use them for training. 

Images from all these datasets are combined into two corresponding directories, which are later used for data preparation and model training. \
Resulting dataset contains 984 fire images and 784 non-fire images.

**2.2 UNet dataset**
- [BoWFire](https://bitbucket.org/gbdi/bowfire-dataset/downloads/) - folder with fire and non-fire images, and folder with fire masks for segmentation training.

UNet model uses both types of images from this dataset and segmentation masks for training. Dataset contains 119 fire images and 107 non-fire images. 

### **3. SETTING UP PROJECT**
- Clone project `git clone <link>`
- Configure virtual environment (example for PyCharm IDE is below)
- Install requirements `pip install -r requirements.txt`
- Download trained models (from [Google Drive](https://drive.google.com/drive/folders/1aSTjtXbVzl8ns48yK4-uO8dzx4ggbZm2?usp=sharing))
- Put models into `models` dir
- Create `image_edition` dir in root of the project

### **4. RUN APP**
- open terminal inside `venv`
- run start script:
    ```bash
    python app.py
    ```

### **5. ADDITIONAL**
**Configure virtual environment (PyCharm)**
- Open in project PyCharm
- Open `Settings` (Ctrl + Alt + S)
- Click on `Project: neural_networks_course_project`
- Click on `Python Interpreter`
- Click on settings icon
- Click on `Add...`
- Configure environment
- Run `pip install -r requirements.txt`

**Useful commands**
- freeze requirements `pip freeze > requirements.txt`

### **6. DEPLOYMENT**
- Used AWS EC2 as a computing resource for our application. 
- Used AWS S3 as a storage for our trained models files.
- Ordered a domain [whereisfire.pp.ua](http://whereisfire.pp.ua) on nic.ua.