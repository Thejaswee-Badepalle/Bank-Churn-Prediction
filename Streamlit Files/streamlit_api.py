import streamlit as st
import pickle
import numpy as np

# Load models
svc_model = pickle.load(open('svc_model1.pkl','rb'))
other_model = pickle.load(open('model.pkl','rb'))

def main():
    st.title('Bank Churn Prediction')

    # Select the model
    model_option = st.selectbox('Select Model', ('SVC Model', 'Logistic Model'))
    
    # Define input fields based on selected model
    if model_option == 'SVC Model':
        st.subheader('SVC Model Features')
        Feature1 = st.number_input('CR_PROD_CNT_IL')
        Feature2 = st.number_input('TURNOVER_PAYM')
        Feature3 = st.number_input('CR_PROD_CNT_CC')
        Feature4 = st.number_input('TRANS_COUNT_SUP_PRC')
        Feature5 = st.number_input('TURNOVER_DYNAMIC_CUR_1M')
        Feature6 = st.number_input('TURNOVER_DYNAMIC_CUR_3M')
        Feature7 = st.number_input('CLNT_SETUP_TENOR')
        Feature8 = st.number_input('REST_AVG_CUR')
        Feature9 = st.number_input('CR_PROD_CNT_TOVR')
        Feature10 = st.number_input('REST_AVG_PAYM')
        placeholder_features = [0] * (59 - 10)  # Adjust based on the actual number of features
        features = np.array([[Feature1, Feature2, Feature3, Feature4, Feature5,
                              Feature6, Feature7, Feature8, Feature9, Feature10] + placeholder_features])
        model = svc_model
    else:
        st.subheader('Logistic Model Features')
        Feature1 = st.number_input('CR_PROD_CNT_IL')
        Feature2 = st.number_input('TURNOVER_PAYM')
        Feature3 = st.number_input('CR_PROD_CNT_CC')
        Feature4 = st.number_input('TRANS_COUNT_SUP_PRC')
        Feature5 = st.number_input('TURNOVER_DYNAMIC_CUR_1M')
        Feature6 = st.number_input('TURNOVER_DYNAMIC_CUR_3M')
        Feature7 = st.number_input('CLNT_SETUP_TENOR')
        Feature8 = st.number_input('REST_AVG_CUR')
        Feature9 = st.number_input('CR_PROD_CNT_TOVR')
        Feature10 = st.number_input('REST_AVG_PAYM')
        placeholder_features = [0] * (36 - 10)  # Adjust based on the actual number of features
        features = np.array([[Feature1, Feature2, Feature3, Feature4, Feature5,
                              Feature6, Feature7, Feature8, Feature9, Feature10] + placeholder_features])
        model = Logistic_model
    
    # Prediction
    if st.button('Predict'):
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')

if __name__ == '__main__':
    main()