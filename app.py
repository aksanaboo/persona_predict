import streamlit as st
import pickle
from preprocess_text import preprocess_text  # Import preprocessing function
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from ast import alias
from ctypes import alignment
from email.mime import image
from textblob import TextBlob

# visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# Load the serialized model
model_filename = 'mbti500_SVCmodel.sav'
with open(model_filename, 'rb') as model_file:
    text_clf = pickle.load(model_file)

logo_image = Image.open('logo1_prev_ui.png')
im = Image.open('human.png')
st.set_page_config(
        page_title="Persona Predict",
        page_icon=im,
        layout="centered",
)

# Use a dictionary to map personality types to colors
personality_colors = {
                                # Analysts
                                'INTJ': '#D2B4DE',
                                'INTP': '#D2B4DE',
                                'ENTJ': '#D2B4DE',
                                'ENTP': '#D2B4DE',

                                # Diplomats
                                'INFJ': '#216f42',
                                'INFP': '#216f42',
                                'ENFJ': '#216f42',
                                'ENFP': '#216f42',

                                # Sentinels
                                'ISTJ': '#A2D9CE',
                                'ISFJ': '#A2D9CE',
                                'ESTJ': '#A2D9CE',
                                'ESFJ': '#A2D9CE',

                                # Explorers
                                'ISTP': '#FFD01D',
                                'ISFP': '#FFD01D',
                                'ESTP': '#FFD01D',
                                'ESFP': '#FFD01D',
                            }

# Define the main Streamlit app
def main():
    st.sidebar.subheader("Navigate App Sections")
    sections = ['Personality Prediction Tool', 'Data Visualization']
    selected_sect = st.sidebar.selectbox("Predict or Visualize:", sections)

    # Display logo image
    col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
    col2.image(logo_image, use_column_width=True)

    if selected_sect == 'Personality Prediction Tool':
        personality_prediction()

    elif selected_sect == 'Data Visualization':
        data_visualization()

def personality_prediction():
    st.title("MBTI Personality Type Predictor")
    st.write("Enter your text to predict your personality type!")

    # User input
    user_input = st.text_area("Enter your text here:")
    if st.button("Predict"):
            if user_input:
                # Preprocess user input
                preprocessed_input = preprocess_text(user_input)

                # Additional length check
                if len(preprocessed_input) < 10:
                    st.error("Invalid text! Enter text with more than 10 letters")
                else:
                    # Make a prediction using the loaded model
                    prediction = text_clf.predict([preprocessed_input])[0]
                    st.write(f"Predicted Personality Type: {prediction}")

                # Get the background color based on the predicted personality from the dictionary
                background_color = personality_colors.get(prediction, '#FFFFFF')  # Default to white if not found
                 # Markdown text with background color
                # Apply background color using st.beta_container
                # Markdown text with background color
                st.markdown(
                    f"""
                    <style>
                        .stApp {{
                            background-color: {background_color};
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )


            #Explanation(Basic Template explanation)

            st.subheader("Explanation")

            if prediction == 'ENFJ':
                st.image('ENFJ.png', width=200)
                st.write("The Teacher: Charismatic and empathetic, ENFJs are natural leaders who inspire and motivate others. They're passionate about helping people reach their potential and creating positive change.")
                st.write("They might work well with INFJs or ENFPs. INFJs offer depth, while ENFPs bring enthusiasm.")
                st.write("For More Details Click here: [ENFJ](https://www.16personalities.com/enfj-personality)")

            elif prediction == 'ENFP':
                st.image('ENFP.png', width=200)
                st.write("The Campaigner: Enthusiastic and imaginative, ENFPs are creative idea generators who inspire those around them. They're driven by their passions and have a natural ability to connect with people.")
                st.write("They might work well with INTPs or INFJs. INTPs can provide logical analysis, while INFJs can contribute emotional depth.")
                st.write("For More Details Click here: [ENFP](https://www.16personalities.com/enfp-personality)")
            elif prediction == 'ENTJ':
                st.image('ENTJ.png', width=200)
                st.write("The Commander: Confident and strategic, ENTJs are born leaders with a strong drive for success. They're effective planners and problem solvers who excel at organizing and executing complex projects.")
                st.write("They might be compatible with ISTJs or ENTJs. ISTJs share a preference for structure, while ENTJs contribute leadership skills.")
                st.write("For More Details Click here: [ENTJ](https://www.16personalities.com/entj-personality)")
            elif prediction == 'ENTP':
                st.image('ENTP.png', width=200)
                st.write("The Debater: Quick-witted and innovative, ENTPs excel at generating ideas and challenging the status quo. They enjoy intellectual debates and thrive in environments that encourage intellectual exploration.")
                st.write("They could collaborate effectively with INTJs or ENFPs. INTJs offer strategic thinking, while ENFPs bring creativity.")
                st.write("For More Details Click here: [ENTP](https://www.16personalities.com/entp-personality)")
            elif prediction == 'ESFJ':
                st.image('ESFJ.png', width=200)
                st.write("The Provider: Warm and supportive, ESFJs are natural caregivers who prioritize the well-being of others. They thrive in social settings and enjoy creating harmony and order in their surroundings.")
                st.write("They might find compatibility with ISFJs or ESTJs. Both combinations value structure and harmony.")
                st.write("For More Details Click here: [ESFJ](https://www.16personalities.com/esfj-personality)")
            elif prediction == 'ESFP':
                st.image('ESFP.png', width=200)
                st.write("The Performer: Fun-loving and outgoing, ESFPs are natural entertainers who enjoy being the center of attention. They bring energy to social situations and have a talent for engaging with others.")
                st.write("They might mesh well with ISFPs or ESFJs. ISFPs share similar values, while ESFJs offer structure.")
                st.write("For More Details Click here: [ESFP](https://www.16personalities.com/esfp-personality)")
            elif prediction == 'ESTJ':
                st.image('ESTJ.png', width=200)
                st.write("The Supervisor: Efficient and organized, ESTJs are natural leaders who value structure and tradition. They excel at managing tasks and people, and their dedication drives them to achieve their goals.")
                st.write("They might be compatible with ISTJs or ENTJs. ISTJs share a preference for structure, while ENTJs contribute leadership skills.")
                st.write("For More Details Click here: [ESTJ](https://www.16personalities.com/estj-personality)")
            elif prediction == 'ESTP':
                st.image('ESTP.png', width=200)
                st.write("The Entrepreneur: Energetic and adaptable, ESTPs thrive in dynamic environments. They're practical problem solvers who love taking risks and are skilled at seizing opportunities.")
                st.write("They might be compatible with ISTPs or ESTJs. ISTPs share practicality, while ESTJs provide organization.")
                st.write("For More Details Click here: [ESTP](https://www.16personalities.com/estp-personality)")
            elif prediction == 'INFJ':
                st.image('INFJ.png', width=200)
                st.write("The Advocate: Empathetic and insightful, INFJs are driven by their ideals and a desire to help others. They're creative, with a deep understanding of human emotions and a vision for positive change.")
                st.write("They might find compatibility with ENFJs or INTJs. ENFJs offer warmth, while INTJs provide strategic thinking.")
                st.write("For More Details Click here: [INFJ](https://www.16personalities.com/infj-personality)")
            elif prediction == 'INFP':
                st.image('INFP.png', width=200)
                st.write("The Mediator: Idealistic and empathetic, INFPs are driven by their values and a desire for authenticity. They're creative and compassionate, often seeking to make a positive impact on the world.")
                st.write("They could work well with ENFJs or ISFJs. ENFJs provide inspiration, and ISFJs offer support.")
                st.write("For More Details Click here: [INFP](https://www.16personalities.com/infp-personality)")
            elif prediction == 'INTJ':
                st.image('INTJ.png', width=200)
                st.write("The Architect: Strategic and independent, INTJs are analytical thinkers with a strong focus on achieving goals. They value competence and are known for their innovative problem-solving skills.")
                st.write("They might find compatibility with ENTJs or INFJs. Both combinations value strategy and depth.")
                st.write("For More Details Click here: [INTJ](https://www.16personalities.com/intj-personality)")
            elif prediction == 'INTP':
                st.image('INTP.png', width=200)
                st.write("The Logician: Analytical and curious, INTPs are deep thinkers who enjoy exploring complex ideas. They're innovative problem solvers, driven by a need to understand the underlying principles of the world.")
                st.write("They could mesh well with ENTPs or INFPs. ENTPs offer idea generation, while INFPs bring a humanistic approach.")
                st.write("For More Details Click here: [INTP](https://www.16personalities.com/intp-personality)")
            elif prediction == 'ISFJ':
                st.image('ISFJ.png', width=200)
                st.write("The Protector: Warm and caring, ISFJs are dependable and compassionate caregivers. They're adept at organizing and providing support, often putting others' needs before their own.")
                st.write("They might find compatibility with ENFJs or INTJs. ENFJs offer warmth, while INTJs provide strategic thinking.")
                st.write("For More Details Click here: [ISFJ](https://www.16personalities.com/isfj-personality)")
            elif prediction == 'ISFP':
                st.image('ISFP.png', width=200)
                st.write("The Artist: Artistic and sensitive, ISFPs are in touch with their emotions and aesthetics. They enjoy expressing themselves through various forms of art and seek harmony in their surroundings.")
                st.write("They might mesh well with ESFJs or ISTPs. ESFJs provide support, while ISTPs offer practical insights.")
                st.write("For More Details Click here: [ISFP](https://www.16personalities.com/isfp-personality)")
            elif prediction == 'ISTJ':
                st.image('ISTJ.png', width=200)
                st.write("The Inspector: Detail-oriented and responsible, ISTJs value tradition and order. They excel at practical tasks, thrive on routine, and take their commitments seriously.")
                st.write("They might find compatibility with ESTJs or ISFJs. Both combinations emphasize structure and detail-oriented work.")
                st.write("For More Details Click here: [ISTJ](https://www.16personalities.com/istj-personality)")
            elif prediction == 'ISTP':
                st.image('ISTP.png', width=200)
                st.write("The Virtuoso: Practical and hands-on, ISTPs are skilled problem solvers who enjoy exploring how things work. They thrive in dynamic environments and excel at finding creative solutions to challenges.")
                st.write("They could collaborate effectively with ESTPs or ISFPs. ESTPs provide action-oriented thinking, while ISFPs bring creativity.")
                st.write("For More Details Click here: [ISTP](https://www.16personalities.com/istp-personality)")
            else:
                st.write("Please enter a valid input")

            st.subheader("Thank You ")

def data_visualization():
    st.title("Data Visualization")
    # sidebar
    st.sidebar.markdown("***")
    st.sidebar.caption("What do they mean?")

    with st.sidebar.expander("16 MBTI Types"):
        st.write('**Analysts**: INTJ, INTP, ENTJ, ENTP')
        st.write('**Diplomats**: INFJ, INFP, ENFJ, ENFP')
        st.write('**Sentinels**: ISTJ, ISFJ, ESTJ, ESFJ')
        st.write('**Explorers**: ISTP, ISFP, ESTP, ESFP')

    with st.sidebar.expander("4 Dimensions"):
        st.write('**IE**: Introvert, Extrovert')
        st.write('**NS**: Intuition, Sensing')
        st.write('**TF**: Thinking, Feeling')
        st.write('**JP**: Judging, Perceiving')

    # Selection Dropdown
    sections = ['Text Analysis', 'Personality Types']
    selected_viz = st.selectbox("Explore the Kaggle Data:", sections)

    # Define a custom color palette
    custom_colors = ["#FFB6C1", "#FF69B4","#FFA07A","#FF6347","#FFD700", "#FFA500","#FF4500", "#FF8C00"]


    # Selection one
    if selected_viz == 'Personality Types':
        st.subheader("The 4 Personality Dimensions")
        # Donut Charts
        fig1 = {
            "data": [
                {"values": [6675, 1999], "labels": ["I", "E"], "domain": {"x": [0.2, 0.5], "y": [0.5, .95]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [7477, 1197], "labels": ["N", "S"], "domain": {"x": [0.51, 0.8], "y": [0.5, .95]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [4693, 3981], "labels": ["T", "F"], "domain": {"x": [0.2, 0.5], "y": [0, 0.45]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [5240, 3434], "labels": ["J", "P"], "domain": {"x": [0.51, 0.8], "y": [0, 0.45]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"}],
            "layout": {"piecolorway": custom_colors, "width":800, 'height':600}
        }

        st.plotly_chart(fig1)

        st.subheader("The 16 Personality Types Distribution")

        # MBTI value counts
        df3 = pd.DataFrame({
            "MBTI Personality Type": ["INTP", "INTJ", "INFJ", "INFP", "ENTP", "ENFP", "ISTP", "ENTJ", "ESTP", "ENFJ", "ISTJ", "ISFP", "ISFJ", "ESTJ", "ESFP", "ESFJ"],
            "Posts Count": [24961,22427,14963,12134,11725,6167,3424,2955,1986,1534,1243,875,650,482,360,181],
        })

        fig2 = px.bar(df3, x="MBTI Personality Type", y="Posts Count", height=600,
                      color_discrete_sequence=['pink'])
        fig2.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig2)

        # Selection two
    elif selected_viz == 'Text Analysis':
        st.subheader("Sentiment & Subjectivity")
        col1, col2 = st.columns(2)
        # Sentiment
        with col1:
            df3 = pd.DataFrame({
                "sentiment": ["Positive", "Negative", "Neutral"],
                "count": [7530, 1127, 17],
            })

            fig3 = px.bar(df3, x="sentiment", y="count", height=400, width=400,
                          color_discrete_sequence=px.colors.qualitative.Pastel2_r)
            fig3.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig3)

        # Subjectivity
        with col2:
            df3 = pd.DataFrame({
                "subjectivity": ["Subjective", "Objective", "Neutral"],
                "count": [7285, 1388, 1],
            })

            fig4 = px.bar(df3, x="subjectivity", y="count", height=400,
                          width=400, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig4.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig4)


if __name__ == "__main__":
    main()
