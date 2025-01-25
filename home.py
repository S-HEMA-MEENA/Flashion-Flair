import subprocess
import sys

# Ensure pip is installed and up-to-date
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Install the required packages from requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

from sentimentalanalysis import load_and_preprocess_data, DataPreparation, DataPreprocessor, ModelTrainer, generate_count_plot, generate_wordclouds, generate_classification_reports
from backend.review import extract_reviews
from sklearn.feature_extraction.text import CountVectorizer
from genai import perform_genai_sentiment_analysis
import streamlit as st
import pandas as pd

def trendzypage():
    st.title('Sentiment Analysis')

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    option = st.sidebar.radio("Choose a section:", ["Extract Reviews", "Perform Sentiment Analysis", "Analyze Sentiments", "Product Quality", "User Review"])

    if option == "Extract Reviews":
        st.subheader('Extract Reviews from Myntra')
        product_id = st.text_input("Enter the Myntra Product ID:")
        if st.button('Extract Reviews'):
            if product_id:
                extract_reviews(product_id)
                st.success('Reviews have been extracted successfully.')
            else:
                st.error('Please enter a product ID.')

    elif option == "Perform Sentiment Analysis":
        st.subheader('Perform Sentiment Analysis using GenAI')
        if st.button('Perform Sentiment Analysis'):
            result_file = perform_genai_sentiment_analysis()
            st.success(f'Sentiment analysis completed. Results saved to {result_file}.')

    elif option == "Analyze Sentiments":
        st.subheader('Sentiment Analysis')

        # Load and preprocess data
        df = load_and_preprocess_data('./dataset.csv')
        data_prep = DataPreparation(df, feature_column='review', label_column='label')
        X_train, X_test, y_train, y_test = data_prep.split_data()

        preprocessor = DataPreprocessor()
        df['review'] = preprocessor.preprocess_dataset(df['review'])

        vect = CountVectorizer(preprocessor=preprocessor.clean)
        X_train_num = vect.fit_transform(X_train)
        X_test_num = vect.transform(X_test)

        model_trainer = ModelTrainer()
        model_trainer.train(X_train_num, y_train)
        y_train_pred, y_test_pred = model_trainer.evaluate(X_train_num, y_train, X_test_num, y_test)

        df['Predicted_Label'] = None
        df.loc[X_train.index, 'Predicted_Label'] = y_train_pred
        df.loc[X_test.index, 'Predicted_Label'] = y_test_pred

        df.to_csv('./dataset.csv', index=False)

        # Streamlit buttons for generating outputs
        st.sidebar.subheader('Generate Outputs')
        plot_option = st.sidebar.radio("Choose an output:", ["Count Plot", "Word Clouds", "Classification Report"])

        if plot_option == "Count Plot":
            if st.button('Generate Count Plot'):
                st.subheader('Distribution of Predicted Labels')
                count_plot_figure = generate_count_plot(df)
                st.pyplot(count_plot_figure)

        elif plot_option == "Word Clouds":
            if st.button('Generate Word Clouds'):
                st.subheader('Word Clouds')
                positive_wordcloud, negative_wordcloud = generate_wordclouds(df)
                
                st.subheader('Positive Reviews Word Cloud')
                st.image(positive_wordcloud.to_image(), caption='Positive Reviews Word Cloud')

                st.subheader('Negative Reviews Word Cloud')
                st.image(negative_wordcloud.to_image(), caption='Negative Reviews Word Cloud')

        elif plot_option == "Classification Report":
            if st.button('Generate Classification Report'):
                st.subheader('Classification Report')
                train_report, test_report = generate_classification_reports(y_train, y_train_pred, y_test, y_test_pred)
                
                st.text('Training Classification Report:')
                st.text(train_report)

                st.text('Test Classification Report:')
                st.text(test_report)

    elif option == "Product Quality":
        st.subheader('Determine Product Quality')
        df = load_and_preprocess_data('./dataset.csv')

        if df.empty:
            st.warning("No data available. Please extract reviews and perform sentiment analysis first.")
        else:
            positive_reviews = df[df['Predicted_Label'] == 1].shape[0]
            negative_reviews = df[df['Predicted_Label'] == 0].shape[0]
            total_reviews = df.shape[0]

            if total_reviews == 0:
                st.warning("No reviews found. Please perform sentiment analysis.")
            else:
                positive_ratio = positive_reviews / total_reviews
                st.write(f"Total Reviews: {total_reviews}")
                st.write(f"Positive Reviews: {positive_reviews}")
                st.write(f"Negative Reviews: {negative_reviews}")
                
                if positive_ratio > 0.6:
                    st.success("The product is generally perceived as good.")
                else:
                    st.error("The product is generally perceived as not good.")

    elif option == "User Review":
        st.subheader('Analyze User Review')
        user_review = st.text_area("Enter your review here:")
        
        if st.button('Analyze Review'):
            if user_review:
                # Preprocess the user review
                preprocessor = DataPreprocessor()
                cleaned_review = preprocessor.preprocess_dataset([user_review])[0]
                
                # Load the trained model
                df = load_and_preprocess_data('./dataset.csv')
                preprocessor = DataPreprocessor()
                vect = CountVectorizer(preprocessor=preprocessor.clean)
                X_train_num = vect.fit_transform(df['review'])
                
                # Transform the user review
                user_review_vectorized = vect.transform([cleaned_review])
                
                model_trainer = ModelTrainer()
                model_trainer.train(X_train_num, df['label'])
                prediction = model_trainer.predict(user_review_vectorized)

                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                st.write(f'The sentiment of your review is: {sentiment}')
            else:
                st.error('Please enter a review.')

if __name__ == "__main__":
    trendzypage()
