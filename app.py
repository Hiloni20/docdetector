# from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel

CLIPProcessor.safety_checker = None

def img2text(uploaded_file):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    if isinstance(uploaded_file, Image.Image):
        image = uploaded_file
    else:
        image = Image.open(uploaded_file)

    array = ["Passport", "Driver License", "Green Card", "401K-statement", "Last-will-and-testament", "life-insurance", "W2-form", "f8889_HSA", "other"]
    inputs = processor(text=array, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)  

    probs = probs.tolist()
    flat_probs = [prob for sublist in probs for prob in sublist]
    max_prob = max(flat_probs)
    index_of_max = flat_probs.index(max_prob)
    
    st.write("Your document has been uploaded successfully. Thanks for submitting your ", array[index_of_max], ".")
    st.write("We'll take care of the rest.")

def main():
    st.set_page_config(page_title='Famiology.docdetector', page_icon='./favicon (2).ico')
    st.header('Famiology Document Detector')
    st.sidebar.image("FamiologyTextLogo.png", use_column_width=True)
    
    with st.sidebar:
        st.header('About App')
        st.header('Smart Document Recognition: Instantly Identify Uploaded Documents')
        st.sidebar.info('Empower your document management process with Smart Document Recognition. This advanced feature swiftly identifies the type of document you upload, making document handling effortless and efficient.')
        st.header('How It Works: ')
        expander = st.expander("See Details")
        expander.write('<ins>Upload Your Document:</ins> Select the document you wish to process using the provided file upload button. \n\n Intelligent Analysis: Our system employs cutting-edge technology to analyze the documents structure, layout, and content. \n\n Automatic Identification: Based on the analysis, Smart Document Recognition accurately identifies the document type, whether its an identification document, real estate document, 401k document or any other document format. \n\n Streamlined Processing: With the document type identified, our platform can seamlessly route it to the appropriate workflow or apply predefined actions, saving you valuable time and effort.')
        st.header('What Problem it Solves?')
        expander = st.expander("See Details")
        expander.write('Efficiency: Instantly recognize document types without manual intervention. \n\n Accuracy: Ensure accurate processing and categorization of documents. \n\n Productivity: Automate document handling workflows for smoother operations.')
        st.header('Value') 
        expander = st.expander("See Details")
        expander.write('eVaults are smart and can support automation of client interactions as well as parallel internal ops process . Saves ops time, cleaner data, nudges for clients as well as for internal staff.')
    
    uploaded_file = st.file_uploader("Choose a file to upload", type=['png', 'jpeg', 'jpg', 'pdf'])
    
    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            img2text(uploaded_file)
        else:
            img2text(uploaded_file)

if __name__ == '__main__':
    main()
