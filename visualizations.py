import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_radar_chart(dimensions):
    """
    Create a radar chart for multi-dimensional sentiment analysis
    
    Args:
        dimensions (dict): Dictionary of dimension scores
        
    Returns:
        plotly.graph_objects.Figure: Radar chart figure
    """
    categories = list(dimensions.keys())
    values = list(dimensions.values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(123, 104, 238, 0.3)',  # Light purple fill
        line=dict(color='rgba(123, 104, 238, 0.8)', width=2),  # Purple line
        name='Sentiment Dimensions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_emotion_distribution(emotions):
    """
    Create a horizontal bar chart for emotion distribution
    
    Args:
        emotions (dict): Dictionary of emotion scores
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    emotions_df = pd.DataFrame({
        'Emotion': list(emotions.keys()),
        'Percentage': list(emotions.values())
    })
    
    # Sort by percentage descending
    emotions_df = emotions_df.sort_values('Percentage', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    colors = {
        'Joy': '#7B68EE',  # Purple
        'Satisfaction': '#5D5CB7',  # Darker purple
        'Interest': '#9370DB',  # Medium purple
        'Trust': '#8A2BE2',  # Violet
        'Fear': '#FF6347',  # Tomato
        'Anger': '#DC143C',  # Crimson
        'Sadness': '#4169E1',  # Royal blue
    }
    
    for i, row in emotions_df.iterrows():
        emotion = row['Emotion']
        percentage = row['Percentage']
        
        fig.add_trace(go.Bar(
            y=[emotion],
            x=[percentage],
            orientation='h',
            marker=dict(
                color=colors.get(emotion, '#7B68EE'),
                line=dict(color='rgb(248, 248, 249)', width=1)
            ),
            text=f"{percentage}%",
            textposition='auto',
            name=emotion,
            showlegend=False
        ))
    
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            domain=[0, 1],
            range=[0, 100],
            tickfont=dict(size=12),
            title=dict(text='%', font=dict(size=12))
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(size=12)
        ),
        margin=dict(l=10, r=10, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        barmode='stack'
    )
    
    return fig

def create_model_comparison(model_results, traditional=True, ai=True, ensemble=True):
    """
    Create a visual comparison of model results
    
    Args:
        model_results (dict): Dictionary of model results
        traditional (bool): Whether traditional NLP was used
        ai (bool): Whether AI models were used
        ensemble (bool): Whether ensemble results were used
        
    Returns:
        plotly.graph_objects.Figure: Model comparison figure
    """
    # Prepare data for visualization
    models = []
    scores = []
    categories = []
    
    if traditional and 'traditional' in model_results:
        models.append('Traditional NLP')
        scores.append(model_results['traditional']['positive'])
        categories.append('Positive')
        
        models.append('Traditional NLP')
        scores.append(model_results['traditional']['negative'])
        categories.append('Negative')
        
        models.append('Traditional NLP')
        scores.append(model_results['traditional']['neutral'])
        categories.append('Neutral')
    
    if ai and 'ai' in model_results:
        models.append('AI Models')
        scores.append(model_results['ai']['positive'])
        categories.append('Positive')
        
        models.append('AI Models')
        scores.append(model_results['ai']['negative'])
        categories.append('Negative')
        
        models.append('AI Models')
        scores.append(model_results['ai']['neutral'])
        categories.append('Neutral')
    
    if ensemble and 'ensemble' in model_results:
        models.append('Ensemble Result')
        scores.append(model_results['ensemble']['positive'])
        categories.append('Positive')
        
        models.append('Ensemble Result')
        scores.append(model_results['ensemble']['negative'])
        categories.append('Negative')
        
        models.append('Ensemble Result')
        scores.append(model_results['ensemble']['neutral'])
        categories.append('Neutral')
    
    # Create dataframe
    df = pd.DataFrame({
        'Model': models,
        'Score': scores,
        'Category': categories
    })
    
    # Create box-like display
    fig = go.Figure()
    
    # Define colors
    model_colors = {
        'Traditional NLP': '#4285F4',  # Blue
        'AI Models': '#7B68EE',  # Purple
        'Ensemble Result': '#34A853'   # Green
    }
    
    # Extract unique models
    unique_models = df['Model'].unique()
    
    # Add text annotations for each model's results
    for model in unique_models:
        model_data = df[df['Model'] == model]
        
        # Extract scores by category
        pos_score = model_data[model_data['Category'] == 'Positive']['Score'].values[0]
        neg_score = model_data[model_data['Category'] == 'Negative']['Score'].values[0]
        neu_score = model_data[model_data['Category'] == 'Neutral']['Score'].values[0]
        
        # Add method information
        method = ""
        if model == 'Traditional NLP' and 'traditional' in model_results:
            method = model_results['traditional'].get('method', 'Basic lexicon-based analysis')
        elif model == 'AI Models' and 'ai' in model_results:
            method = model_results['ai'].get('method', 'Hugging Face sentiment analysis')
        elif model == 'Ensemble Result' and 'ensemble' in model_results:
            method = model_results['ensemble'].get('method', 'Combined with contextual weighting')
        
        # Create the box layout
        model_index = list(unique_models).index(model)
        x_pos = model_index * 0.33
        
        # Add rectangle shape
        fig.add_shape(
            type="rect",
            x0=x_pos, y0=0,
            x1=x_pos + 0.32, y1=1,
            fillcolor="white",
            line=dict(color=model_colors.get(model, "#000"), width=2),
        )
        
        # Add model name
        fig.add_annotation(
            x=x_pos + 0.16, y=0.95,
            text=f"<b>{model}</b>",
            showarrow=False,
            font=dict(size=14, color="#333")
        )
        
        # Add scores
        y_positions = [0.8, 0.7, 0.6, 0.4]
        
        # Positive
        fig.add_annotation(
            x=x_pos + 0.05, y=y_positions[0],
            text="Positive:",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="left"
        )
        fig.add_annotation(
            x=x_pos + 0.28, y=y_positions[0],
            text=f"{pos_score:.2f}",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="right"
        )
        
        # Negative
        fig.add_annotation(
            x=x_pos + 0.05, y=y_positions[1],
            text="Negative:",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="left"
        )
        fig.add_annotation(
            x=x_pos + 0.28, y=y_positions[1],
            text=f"{neg_score:.2f}",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="right"
        )
        
        # Neutral
        fig.add_annotation(
            x=x_pos + 0.05, y=y_positions[2],
            text="Neutral:",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="left"
        )
        fig.add_annotation(
            x=x_pos + 0.28, y=y_positions[2],
            text=f"{neu_score:.2f}",
            showarrow=False,
            font=dict(size=12, color="#333"),
            xanchor="right"
        )
        
        # Method
        fig.add_annotation(
            x=x_pos + 0.16, y=y_positions[3],
            text=f"Method: {method}",
            showarrow=False,
            font=dict(size=11, color="#666"),
            xanchor="center"
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, len(unique_models) * 0.33 + 0.01]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 1]
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_sentiment_breakdown_table(df):
    """
    Create a table visualization for sentiment breakdown by sentence
    
    Args:
        df (pandas.DataFrame): DataFrame with sentence analysis
        
    Returns:
        plotly.graph_objects.Figure: Table figure
    """
    # Define colors for sentiment
    sentiment_colors = {
        "Very Positive": "#2E8B57",
        "Positive": "#4CAF50",
        "Neutral": "#FF9800",
        "Negative": "#F44336",
        "Very Negative": "#B71C1C"
    }
    
    # Create table
    fig = go.Figure(data=[go.Table(
        columnwidth=[6, 2, 2, 3],
        header=dict(
            values=['<b>SENTENCE</b>', '<b>SENTIMENT</b>', '<b>CONFIDENCE</b>', '<b>KEY ENTITIES</b>'],
            line_color='white',
            fill_color='#f0f0f0',
            align='left',
            font=dict(size=13, color='#333')
        ),
        cells=dict(
            values=[
                df['sentence'],
                df['sentiment'],
                df['confidence'],
                df['entities']
            ],
            line_color='white',
            fill_color=[
                ['white'] * len(df),
                [sentiment_colors.get(s, "#9E9E9E") for s in df['sentiment']],
                ['#f9f9f9'] * len(df),
                ['white'] * len(df)
            ],
            align='left',
            font=dict(size=12, color=['#333', 'white', '#333', '#333']),
            height=30
        )
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50 + (len(df) * 35),  # Dynamic height based on number of rows
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig
