import streamlit as st
import pandas as pd
import numpy as np
from sentiment_analyzer import analyze_sentiment, get_sentiment_breakdown
from visualizations import (
    create_radar_chart,
    create_emotion_distribution,
    create_model_comparison,
    create_sentiment_breakdown_table
)
from utils import get_sentiment_label, get_confidence_color, get_domain_options

# Page configuration
st.set_page_config(
    page_title="SentiScape - Advanced Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom header with logo and help button
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="20" cy="20" r="20" fill="#7C4DFF" opacity="0.9"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="white" font-size="16px">S</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown('<h2 style="color: #333; margin-bottom: 0;">SentiScape</h2>', unsafe_allow_html=True)
with col3:
    help_button = st.button("Help")

if help_button:
    st.info("""
    **SentiScape Help**
    
    SentiScape is an advanced sentiment analysis tool that helps you analyze the emotional tone of text.
    
    **How to use:**
    1. Select an analysis domain from the dropdown
    2. Enter or paste the text you want to analyze
    3. Select at least one analysis method
    4. Click "Analyze Sentiment" to see detailed results
    
    For best results, provide at least 3 sentences for analysis.
    """)

# Main content
st.markdown('<h3 style="color: #333;">Analyze Text Sentiment</h3>', unsafe_allow_html=True)

# Analysis domain selection
st.markdown('<p style="color: #555; font-size: 16px; margin-bottom: 8px;">Analysis Domain</p>', unsafe_allow_html=True)
domain_options = get_domain_options()
selected_domain = st.selectbox("", domain_options, index=0, label_visibility="collapsed")

# Text input area
st.markdown('<p style="color: #555; font-size: 16px; margin-bottom: 8px;">Enter text for analysis</p>', unsafe_allow_html=True)
user_text = st.text_area(
    "", 
    placeholder="Type or paste text here for sentiment analysis...",
    height=150,
    label_visibility="collapsed"
)

# Usage tip
st.caption("For best results, enter at least 3 sentences.")

# Analysis method selection
col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
with col1:
    traditional_nlp = st.checkbox("Traditional NLP", value=True)
with col2:
    ai_models = st.checkbox("AI Models", value=True)
with col3:
    ensemble_results = st.checkbox("Ensemble Results", value=True)
with col4:
    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=False)

# Validate if any analysis method is selected
if not any([traditional_nlp, ai_models, ensemble_results]) and analyze_button:
    st.warning("Please select at least one analysis method.")
    analyze_button = False

# Process analysis when button is clicked
if analyze_button and user_text.strip():
    # Check minimum text length
    if len(user_text.split()) < 6:
        st.warning("Please provide more text for a comprehensive analysis (at least 3 sentences recommended).")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Get analysis results
            results = analyze_sentiment(
                user_text, 
                domain=selected_domain,
                use_traditional=traditional_nlp,
                use_ai=ai_models,
                use_ensemble=ensemble_results
            )
            
            # Display analysis results
            st.markdown("## Analysis Results")
            st.caption(user_text[:300] + "..." if len(user_text) > 300 else user_text)
            
            # Overall sentiment metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_label = get_sentiment_label(results['overall_sentiment'])
                st.markdown(f"""
                <div style="display: inline-block; background-color: {get_confidence_color(sentiment_label)}; 
                color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">
                    ● Overall: {sentiment_label}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="display: inline-block; background-color: #7B68EE; 
                color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">
                    ● Confidence: {results['confidence']}%
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if results.get('mixed_sentiments', False):
                    st.markdown(f"""
                    <div style="display: inline-block; background-color: #FFA500; 
                    color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">
                        ● Mixed Sentiments
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualization section
            col1, col2 = st.columns([3, 3])
            
            with col1:
                st.markdown("### MULTI-DIMENSIONAL SENTIMENT")
                radar_chart = create_radar_chart(results['dimensions'])
                st.plotly_chart(radar_chart, use_container_width=True)
                
                st.markdown("### EMOTION DISTRIBUTION")
                emotion_chart = create_emotion_distribution(results['emotions'])
                st.plotly_chart(emotion_chart, use_container_width=True)
                
            with col2:
                st.markdown("### CONTEXTUAL ANALYSIS")
                st.markdown("""
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                    <p><strong>Domain Context</strong><br/>
                    <span style="color: #555;">{}</span></p>
                    
                    <p><strong>Key Aspects</strong></p>
                    
                    <p><strong>Cultural Context</strong><br/>
                    <span style="color: #555;">Analysis performed in general context with Hugging Face models.</span></p>
                    
                    <p><strong>Audience Perception</strong><br/>
                    <div style="margin-top: 5px;">
                        <div style="width: {}%; background-color: #7B68EE; height: 10px; border-radius: 5px;"></div>
                        <p style="margin-top: 5px; font-size: 14px;">{}% estimated positive perception by target audience</p>
                    </div>
                    </p>
                </div>
                """.format(selected_domain.lower(), results['audience_perception'], results['audience_perception']), 
                unsafe_allow_html=True)
            
            # Model comparison section
            st.markdown("### MODEL COMPARISON")
            model_comparison = create_model_comparison(
                results['model_results'],
                traditional_nlp,
                ai_models,
                ensemble_results
            )
            st.plotly_chart(model_comparison, use_container_width=True)
            
            # Sentiment breakdown
            st.markdown("### SENTIMENT BREAKDOWN")
            breakdown_df = get_sentiment_breakdown(user_text)
            breakdown_table = create_sentiment_breakdown_table(breakdown_df)
            st.plotly_chart(breakdown_table, use_container_width=True)
            
            # Historical analysis section
            st.markdown("### HISTORICAL ANALYSIS")
            col1, col2 = st.columns([6, 1])
            with col1:
                st.caption("Recent Analyses")
            with col2:
                st.button("View All History", disabled=True)
elif analyze_button and not user_text.strip():
    st.warning("Please enter some text to analyze.")
