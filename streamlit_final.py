# import warnings
# warnings.filterwarnings("ignore")

# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import os

# # Load the trained model
# model_path = "waste_classifier_mobilenet.h5"
# model = load_model(model_path)

# # Define the class labels
# categories = [
#     'AEROSOL CANS', 'ALUMINUM FOOD CAN', 'ALUMINUM SODA CAN', 'CARDBOARD BOX', 'CLOTHES',  
#     'COFFEE GROUND', 'DISPOSABLE PLASTIC CUTLERY', 'EGGSHELL', 'FOOD WASTE', 'GLASS BEVERAGE BOTTLE',
#     'GLASS COSMETIC CONTAINERS', 'GLASS FOOD JAR', 'MAGAZINES', 'NEWSPAPER',
#     'OFFICE PAPER', 'PAPER CUP', 'PLASTIC CUP LID', 'PLASTIC DETERGENT BOTTLE', 'PLASTIC FOOD CONTAINER',
#     'PLASTIC SHOPPING BAG', 'PLASTIC SODA BOTTLE', 'PLASTIC STRAW', 'PLASTIC TRASH BAG', 'PLASTIC WATER BOTTLE',
#     'SHOES', 'STEEL FOOD CANS', 'STYROFORM CUPS', 'STYROFORM FOOD CONTAINERS', 'TEA BAGS'
# ]

# # Function to preprocess the uploaded image
# def prepare_image(image_path):
#     img = load_img(image_path, target_size=(128, 128))  # Resize image to the required size
#     img_array = img_to_array(img)  # Convert to array
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape
#     img_array = img_array / 255.0  # Normalize the image
#     return img_array

# # Streamlit interface
# st.title("Waste Classification for Smart Bins using CNN")

# # File uploader for image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

# # Check if a file is uploaded
# if uploaded_file is not None:
#     # Save the uploaded file temporarily
#     temp_file_path = os.path.join("temp_dir", uploaded_file.name)
#     with open(temp_file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     image_for_model = prepare_image(temp_file_path)

#     # Predict the category
#     prediction = model.predict(image_for_model)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     predicted_label = categories[predicted_class]

#     # Display the prediction
#     st.write(f"The image belongs to the category: **{predicted_label}**")

#     # Clean up temporary files after use
#     os.remove(temp_file_path)
# else:
#     st.write("Please upload an image file to get started.")



















import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the trained model
model_path = "waste_classifier_mobilenet.h5"
model = load_model(model_path)

# Define the class labels and related points
categories = [
    'AEROSOL CANS', 'ALUMINUM FOOD CAN', 'ALUMINUM SODA CAN', 'CARDBOARD BOX', 'CLOTHES',  
    'COFFEE GROUND', 'DISPOSABLE PLASTIC CUTLERY', 'EGGSHELL', 'FOOD WASTE', 'GLASS BEVERAGE BOTTLE',
    'GLASS COSMETIC CONTAINERS', 'GLASS FOOD JAR', 'MAGAZINES', 'NEWSPAPER',
    'OFFICE PAPER', 'PAPER CUP', 'PLASTIC CUP LID', 'PLASTIC DETERGENT BOTTLE', 'PLASTIC FOOD CONTAINER',
    'PLASTIC SHOPPING BAG', 'PLASTIC SODA BOTTLE', 'PLASTIC STRAW', 'PLASTIC TRASH BAG', 'PLASTIC WATER BOTTLE',
    'SHOES', 'STEEL FOOD CANS', 'STYROFORM CUPS', 'STYROFORM FOOD CONTAINERS', 'TEA BAGS'
]

# Points for each category
category_details = {
    'AEROSOL CANS': ['You can recycle aerosol cans if they are empty.', 'Aerosol cans that are not empty can be hazardous', 'To check if the can is empty, hold it near a rag and spray until nothing comes out.'],
    
    'ALUMINUM FOOD CAN': ['Aluminum food cans can be recycled, it is infinitely recyclable', 'Aluminum is 100% recyclable and can be recycled endlessly without losing quality.'],
    
    'ALUMINUM SODA CAN': ['Aluminum food cans can be recycled, it is infinitely recyclable', 'Aluminum is 100% recyclable and can be recycled endlessly without losing quality.'],
    
    'CARDBOARD BOX': ['Cardboard boxes are easily recyclable.', 'Flatten boxes to conserve space'],
    'CLOTHES': ['Clothing can be recycled.', 'Textiles can take years to decompose','Recycling clothing is an important way to reduce the environmental impact of clothing and textiles.'],
    
    'COFFEE GROUND': ['Coffee grounds can be recycled in many 				ways.','Good for composting', 'Rich in nitrogen'],
    
    'DISPOSABLE PLASTIC CUTLERY': ['It is not recyclable, can only be landfilled or incinerated.'],
    
    'EGGSHELL': ['Egg shells are NOT recyclable','Great for composting', 'Rich in calcium for plants'],
    
    'FOOD WASTE': ['Food waste can be recycled in several ways','Compostable', 'Avoid sending to landfill to reduce methane emissions'],
    
    'GLASS BEVERAGE BOTTLE': ['glass beverage bottles are 100% 					recyclable and can be recycled indefinitely without losing quality.'],

    'GLASS COSMETIC CONTAINERS': ['glass cosmetic containers like perfume bottles, jars for face creams, and decorative jars are recyclable.'],

    'GLASS FOOD JAR': ['Glass food jars are recyclable.', '100% recyclable and can be recycled indefinitely without losing quality.'],

    'MAGAZINES': ['Yes, it can be recycled.'],

    'NEWSPAPER': ['Yes, it can be recycled.'],

    'OFFICE PAPER': ['Yes, it can be recycled.'],

    'PAPER CUP': ['PAPER CUP'],

    'PLASTIC CUP LID':['Some plastic cup lids are recyclable, but it depends on the type of plastic and the type of cup.'],

    'PLASTIC DETERGENT BOTTLE': ['You can recycle plastic detergent bottles.'],

    'PLASTIC FOOD CONTAINER':['Many plastic food containers can be recycled.'],

    'PLASTIC SHOPPING BAG':['plastic shopping bags can be recycled.'],

    'PLASTIC SODA BOTTLE': ['Most plastic soda bottles can be recycled','Most plastic soda bottles are made from polyethylene terephthalate (PET), also known as type-1 plastic, which is one of the most recyclable plastics.'],

    'PLASTIC STRAW':['Plastic straws are generally not recyclable and should be thrown away in the trash.','Plastic straws are too small to be sorted by recycling machines and can fall through the screens.'],

    'PLASTIC TRASH BAG':['most plastic trash bags can be recycled, but you can not put them in your curbside recycling bin'],

    'PLASTIC WATER BOTTLE':['Plastic water bottles can be recycled','Plastic bottles are collected, shredded, melted, and reformed into pellets. These pellets are then used to make new plastic goods, including more water bottles.'],

    'SHOES':['Shoes can be recycled at some recycling centers, charity shops, or shoe banks','Some recycling centers have shoe banks or recycle bins where you can drop off your shoes.'],

    'STEEL FOOD CANS':['Steel food cans can be recycled','Steel cans can be recycled without losing quality and can be recycled over and over again.'],

    'STYROFORM CUPS':['Styrofoam cups are not generally recyclable.', 'Styrofoam is made from expanded polystyrene (EPS), which is more than 90% air and only 2% plastic. This makes it lightweight and bulky, and difficult to recycle.'],

    'STYROFORM FOOD CONTAINERS':['Styrofoam food containers are not commonly recyclable and are usually disposed of in the trash.'],

    'TEA BAGS':['Tea bags containing polypropylene are not totally biodegradable, although they can be recycled in your food caddy. If you want to compost them at home, you can sift the plastic out later.'],
}

# Function to preprocess the uploaded image
def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize image to the required size
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit interface
st.title("Waste Classification for Smart Bins using CNN")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp_dir", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_for_model = prepare_image(temp_file_path)

    # Predict the category
    prediction = model.predict(image_for_model)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = categories[predicted_class]

    # Display the prediction
    st.write(f"The image belongs to the category: **{predicted_label}**")

    # Display additional points for the predicted category
    if predicted_label in category_details:
        st.write("Additional Information:")
        for point in category_details[predicted_label]:
            st.write(f"- {point}")

    # Clean up temporary files after use
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file to get started.")