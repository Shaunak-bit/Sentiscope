import sqlite3
import json
from datetime import datetime

class SentimentDatabase:
    def __init__(self, db_path='sentiment_analysis.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            with open('schema_fixed.sql', 'r') as schema_file:
                conn.executescript(schema_file.read())

    def save_analysis(self, text_content, overall_scores, sentence_analyses):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert main analysis result
            cursor.execute("""
                INSERT INTO analysis_results 
                (text_content, overall_sentiment_score, compound_score, 
                positive_score, negative_score, neutral_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                text_content,
                overall_scores.get('sentiment', overall_scores.get('compound', 0)),
                overall_scores.get('compound', 0),
                overall_scores.get('positive', overall_scores.get('pos', 0)),
                overall_scores.get('negative', overall_scores.get('neg', 0)),
                overall_scores.get('neutral', overall_scores.get('neu', 0)),
                json.dumps(overall_scores.get('metadata', {}))
            ))
            
            analysis_id = cursor.lastrowid
            
            # Insert sentence-level analyses
            for idx, sentence_data in enumerate(sentence_analyses):
                cursor.execute("""
                    INSERT INTO sentence_analysis
                    (analysis_id, sentence_text, sentence_index, sentiment_score,
                    compound_score, positive_score, negative_score, neutral_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    sentence_data['text'],
                    idx,
                    sentence_data.get('sentiment', sentence_data.get('compound', 0)),
                    sentence_data.get('compound', 0),
                    sentence_data.get('positive', sentence_data.get('pos', 0)),
                    sentence_data.get('negative', sentence_data.get('neg', 0)),
                    sentence_data.get('neutral', sentence_data.get('neu', 0))
                ))
            
            conn.commit()
            return analysis_id

    def get_analysis(self, analysis_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get main analysis
            analysis = cursor.execute(
                "SELECT * FROM analysis_results WHERE id = ?", 
                (analysis_id,)
            ).fetchone()
            
            if not analysis:
                return None
            
            # Get sentence analyses
            sentences = cursor.execute(
                "SELECT * FROM sentence_analysis WHERE analysis_id = ? ORDER BY sentence_index",
                (analysis_id,)
            ).fetchall()
            
            return {
                'analysis': dict(analysis),
                'sentences': [dict(sentence) for sentence in sentences]
            }

    def get_all_analyses(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            analyses = cursor.execute("SELECT * FROM analysis_results ORDER BY timestamp DESC").fetchall()
            return [dict(analysis) for analysis in analyses]

    def delete_analysis(self, analysis_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete sentence analyses first due to foreign key constraint
            cursor.execute("DELETE FROM sentence_analysis WHERE analysis_id = ?", (analysis_id,))
            # Delete main analysis
            cursor.execute("DELETE FROM analysis_results WHERE id = ?", (analysis_id,))
            conn.commit()
            return cursor.rowcount > 0
