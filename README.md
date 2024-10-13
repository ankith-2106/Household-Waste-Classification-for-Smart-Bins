# Project title:- Waste Classification for Smart Bins

# 1. *Data Collection*

    # Task: I gathered images of household waste of 29 different categories, with around 700-1200 images per category.
          The categories for my project were as folows,
            AEROSOL CANS
            ALUMINUM FOOD CAN
            ALUMINUM SODA CAN
            CARDBOARD BOX
            CLOTHES
            COFFEE GROUND
            DISPOSABLE PLASTIC CUTLERY
            EGGSHELL
            FOOD WASTE
            GLASS BEVERAGE BOTTLE
            GLASS COSMETIC CONTAINERS
            GLASS FOOD JAR
            MAGAZINES
            NEWSPAPER
            OFFICE PAPER
            PAPER CUP
            PLASTIC CUP LID
            PLASTIC DETERGENT BOTTLE
            PLASTIC FOOD CONTAINER
            PLASTIC SHOPPING BAG
            PLASTIC SODA BOTTLE
            PLASTIC STRAW
            PLASTIC TRASH BAG
            PLASTIC WATER BOTTLE
            SHOES
            STEEL FOOD CANS
            STYROFOAM CUPS
            STYROFOAM FOOD CONTAINERS
            TEA BAGS
  
    *Next Steps*: Preprocessing the dataset, ensuring that the images of each categories have correct images. Secondly removing any 
                dodgy images and thirdly removing all those images whose size is less than 5 kb.
                Images with extension like "jpeg", "jpg", "bmp" and "png" were only selected for training process.

# 2. *Data Preprocessing*

    *Image Resizing*: Resize images to a uniform size (e.g., 128x128 pixels) to reduce memory usage.
  
    *Data Augmentation*: Techniques like shear, zoom, and horizontal flips are applied to enhance generalization and prevent overfitting.
  
    *Normalization*: The pixel values are rescaled to the range [0, 1] using ImageDataGenerator to improve the model's training.

# 3. *Model Selection and Training*

    *Base Model*: I choose MobileNetV2, reason being MobileNetV2 is lightweight and it is an advanced deep learning architecture designed for efficient image recognition
                and classification tasks. Pre-trained on ImageNet, which can be fine-tuned on my waste classification dataset, it is a lightweight model that 
                balances accuracy and speed. The base model's layers are frozen to prevent updating pre-trained weights. Custom layers (global average pooling, 
                dense, and dropout) are added for classification.

    *Custom Layers*: Added a GlobalAveragePooling2D layer to reduce the dimensionality. Added a Dense(512, relu) layer for classification. Added a Dropout layer 
                   to mitigate overfitting. Final Dense layer outputs 29 categories using a softmax activation function.

    *Training Strategy*: Freezing the MobileNetV2 layers ensures computational efficiency. Optimized with Adam optimizer and categorical cross-entropy loss.
                       Trained with early stopping to prevent overfitting.

# 4. *Model Evaluation*

    *Metrics*: During training, I have tracked accuracy.

    *Validation*: I am using a validation set during training, this ensures that the model is evaluated on unseen data after each epoch, helping me to monitor its ability to generalize.

    *Performance*: The balance between inference time and accuracy is crucial. While the MobileNetV2 architecture is optimized for speed and low computational 
                 requirements, I should also monitor inference time during deployment.

# 5. *Deployment*

    *Streamlit Interface*: I designed a user-friendly interface where users upload an image, and the system predicts its category. The prediction, and also that particular waste 
                         can be recycled or not and if it can be recycled how it can be recycled, will be displayed.

    *Backend*: The MobileNetV2 model will be loaded into the Streamlit app, and the uploaded image will be preprocessed before classification.

# 6. *Note*
    a. Code till model evaluation has been done in Final_code.ipynb file.
    b. Code for streamlit has been given in streamlit_final.py file.
