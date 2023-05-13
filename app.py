import streamlit as sl
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import random
import torch
from PIL import Image

#model loaded from huggingface
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#for generating different multiple captions
def generate_caption(image, num_captions):
    images = []
    i_image=image
    #to change the image to rgb format 
    if i_image.mode != "RGB":
        i_image=i_image.convert(mode="RGB")
    images.append(i_image)
    #to extract the features from the input/uploaded image
    pixel_values=feature_extractor(images=i_image, return_tensors="pt").pixel_values.to(device)
    captions=[]
    for i in range(num_captions):
       seed=random.randint(0,100000)
       torch.manual_seed(seed)
       output_ids=model.generate(pixel_values, do_sample=True, num_beams=1, temperature=1.0, max_length=50, pad_token_id=tokenizer.pad_token_id)
       caption=tokenizer.decode(output_ids[0], skip_special_tokens=True)
       captions.append(caption)
    return captions

   

def main():

    sl.title("Image Caption Generator")
    #to upload the image
    image_file = sl.file_uploader("Upload Images" , type=["png","jpg","jpeg"])
    # number of captions according to the user
    num_captions=int(sl.number_input("Enter the number of captions required ", min_value=1, max_value=5))
    if image_file is not None:
        #to see the image
      
        sl.image(image_file)
        if sl.button('Generate Captions'):
            captions= generate_caption(Image.open(image_file), num_captions)
            for caption in captions:
                sl.write(caption)
          
    
if __name__ =='__main__':
    main()