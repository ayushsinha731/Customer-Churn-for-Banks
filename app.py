import streamlit as st
import tensorflow as tf
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# Load The model
model = tf.keras.models.load_model('model.h5')
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])
# Load the encoders and scalars
with open('geo_ohe.pkl','rb') as file:
    geo_ohe = pkl.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pkl.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pkl.load(file)

#streamlit app
st.title('Customer Disloyalty Prediction')

# User input

geography = st.selectbox('Geography', [string.split('_')[1] for string in geo_ohe.get_feature_names_out()])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',1,10)
num_of_products = st.slider('Number of products',1,10)
has_cr_card=st.selectbox('Credit Card',[0,1])
is_active_member=st.selectbox('Active Member',[0,1])

#Pre-process the input

input_data = pd.concat([
    pd.DataFrame({
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        # 'Geography':[geography],
        'Balance': [balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary]
    }).reset_index(drop=True),
    pd.DataFrame(
        geo_ohe.transform([[geography]]).toarray(),
        columns=geo_ohe.get_feature_names_out(['Geography'])
    )],
    axis=1
)
# Scale the data
input_data_scaled = scaler.transform(input_data)

# Predict 
prediction  = model.predict(input_data_scaled)
prediction_probablity = prediction[0][0]
if prediction_probablity > 0.5:
    st.write("This customer likely to leave us.")
else:
    st.write("This customer is not likely to leave us.")