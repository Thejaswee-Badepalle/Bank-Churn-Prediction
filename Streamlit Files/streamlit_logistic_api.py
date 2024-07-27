import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

def main():
    st.title('Bank Churn Prediction')

    #input Variables
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
    
    # Assume the remaining features are placeholders
    placeholder_features = [0] * (36 - 10)  # Placeholder values for the missing features
    
    #prediction code
    if st.button('Predict'):
        features = np.array([[Feature1, Feature2, Feature3, Feature4, Feature5,
                              Feature6,Feature7,Feature8,Feature9,Feature10]+ placeholder_features])
    

    
        # Make the prediction
        prediction = model.predict(features)
    
        # Display the result
        if prediction[0] == 1:
            st.success('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')

if __name__ == '__main__':
    main()