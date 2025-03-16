# Step 1: Install required libraries
#!pip install gradio vaderSentiment pandas matplotlib nltk

# Step 2: Import necessary libraries
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from nltk.tokenize import sent_tokenize

# Step 3: Ensure NLTK resources are downloaded
try:
    # Explicitly download the 'punkt' tokenizer
    nltk.download('punkt')
except Exception as e:
    raise ValueError(f"Failed to download NLTK resources: {str(e)}")

# Step 4: Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 5: Define functions for processing and analysis
def analyze_text(user_input):
    """
    Function to analyze sentiment for text entered in the text box.
    """
    try:
        sentence_df, doc_df, error = process_corpus(user_input.splitlines())
        if error:
            return pd.DataFrame(), pd.DataFrame(), error
        return sentence_df, doc_df, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {str(e)}"

def analyze_file(file):
    """
    Function to analyze sentiment for text uploaded via a .txt file.
    """
    try:
        # Attempt to read the file with UTF-8 encoding
        with open(file.name, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()  # Each line is treated as a separate document
        sentence_df, doc_df, error = process_corpus(corpus)
        if error:
            return pd.DataFrame(), pd.DataFrame(), error
        return sentence_df, doc_df, None
    except UnicodeDecodeError:
        # If UTF-8 fails, try reading with another encoding (e.g., latin-1)
        try:
            with open(file.name, 'r', encoding='latin-1') as f:
                corpus = f.read().splitlines()
            sentence_df, doc_df, error = process_corpus(corpus)
            if error:
                return pd.DataFrame(), pd.DataFrame(), error
            return sentence_df, doc_df, None
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), f"Error processing file: {str(e)}"
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Error processing file: {str(e)}"

def process_corpus(corpus):
    """
    Process a list of documents (corpus) and analyze sentiment for each sentence in each document.
    Compute document-level sentiment scores.
    """
    results = []
    doc_scores = []
    
    for doc_idx, doc in enumerate(corpus, start=1):
        if doc.strip():  # Ignore empty documents
            try:
                # Tokenize document into sentences using NLTK's sent_tokenize
                sentences = sent_tokenize(doc.strip())
                
                doc_compound_score = 0
                
                for sent_idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():  # Ensure sentence is not empty
                        senti_scores = analyzer.polarity_scores(sentence)
                        senti_scores_rounded = {k: round(v, 3) for k, v in senti_scores.items()}
                        
                        # Add sentence-level results
                        results.append({
                            'doc_ID': doc_idx,
                            'sent_ID': sent_idx,
                            'sentence': sentence,
                            'compound': senti_scores_rounded['compound'],
                            'neg': senti_scores_rounded['neg'],
                            'neu': senti_scores_rounded['neu'],
                            'pos': senti_scores_rounded['pos']
                        })
                        
                        # Accumulate compound score for document-level analysis
                        doc_compound_score += senti_scores_rounded['compound']
                
                # Add document-level results
                doc_scores.append({
                    'doc_ID': doc_idx,
                    'doc_senti_score': round(doc_compound_score, 3)
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error processing document {doc_idx}: {str(e)}"
    
    # Convert results to DataFrames
    sentence_df = pd.DataFrame(results)
    doc_df = pd.DataFrame(doc_scores)
    
    return sentence_df, doc_df, None

# Step 6: Define the function to generate the plot
def generate_plot(doc_df):
    """
    Generate a line plot of document-level sentiment scores over document ID.
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot document-level sentiment scores
        plt.plot(doc_df['doc_ID'], doc_df['doc_senti_score'], marker='o', linestyle='-', color='blue', label="Document Sentiment")
        
        # Add a horizontal line at y=0 for neutral sentiment
        plt.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label="Neutral Level")
        
        # Add labels, legend, and title
        plt.title("Document-Level Sentiment Scores", fontsize=14)
        plt.xlabel("Document ID", fontsize=12, fontweight='bold')
        plt.ylabel("Document Sentiment Score", fontsize=12)
        plt.xticks(doc_df['doc_ID'])  # Show all document IDs on the x-axis
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to a temporary file
        plot_path = "doc_sentiment_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    except Exception as e:
        raise ValueError(f"Error generating plot: {str(e)}")

# Step 7: Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Sentiment Analysis App with Text Input and File Upload")
    
    # Tab 1: Text Input and Results
    with gr.Tab("Text Input"):
        with gr.Row():
            text_input = gr.Textbox(label="Enter your text here", lines=5, placeholder="Type your text...")
            analyze_button_text = gr.Button("Analyze Sentiment")
        
        sentiment_results_text = gr.Dataframe(label="Sentence-Level Sentiment Analysis Results")
        doc_sentiment_results_text = gr.Dataframe(label="Document-Level Sentiment Analysis Results")
        plot_output_text = gr.Image(label="Document Sentiment Plot", type="filepath")
        error_message_text = gr.Textbox(label="Error Message", interactive=False)
        
        def perform_analysis_text(user_input):
            """
            Perform sentiment analysis and generate plot for text input.
            """
            try:
                sentence_df, doc_df, error = analyze_text(user_input)
                if error:
                    return pd.DataFrame(), pd.DataFrame(), None, error
                plot_path = generate_plot(doc_df)
                return sentence_df, doc_df, plot_path, None
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), None, f"Unexpected error: {str(e)}"
        
        analyze_button_text.click(
            perform_analysis_text,
            inputs=text_input,
            outputs=[sentiment_results_text, doc_sentiment_results_text, plot_output_text, error_message_text]
        )
    
    # Tab 2: File Upload and Results
    with gr.Tab("File Upload"):
        file_input = gr.File(label="Upload a .txt file (each line is a document)")
        analyze_button_file = gr.Button("Analyze Sentiment")
        
        sentiment_results_file = gr.Dataframe(label="Sentence-Level Sentiment Analysis Results")
        doc_sentiment_results_file = gr.Dataframe(label="Document-Level Sentiment Analysis Results")
        plot_output_file = gr.Image(label="Document Sentiment Plot", type="filepath")
        error_message_file = gr.Textbox(label="Error Message", interactive=False)
        
        def perform_analysis_file(file):
            """
            Perform sentiment analysis and generate plot for file upload.
            """
            try:
                sentence_df, doc_df, error = analyze_file(file)
                if error:
                    return pd.DataFrame(), pd.DataFrame(), None, error
                plot_path = generate_plot(doc_df)
                return sentence_df, doc_df, plot_path, None
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), None, f"Unexpected error: {str(e)}"
        
        analyze_button_file.click(
            perform_analysis_file,
            inputs=file_input,
            outputs=[sentiment_results_file, doc_sentiment_results_file, plot_output_file, error_message_file]
        )

# Step 8: Launch the app
demo.launch()
