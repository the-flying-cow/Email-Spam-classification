import pandas as pd
import numpy as np
import re

# patterns to look for
URL_PATTERN = r'https?://\S+|www\.\S+'
REPLY_PATTERN = r'(re:|reply|respond)'
FORWARD_PATTERN = r'(fwd:|fw:|forward)'
HTML_PATTERN = r'<[^>]+>'


def _safe_mean_word_length(text: str) -> float:
    words = text.split()
    return float(np.mean([len(word) for word in words])) if words else 0.0


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.drop(
        columns=[
            'Mime-Version',
            'Content-Transfer-Encoding',
            'Mail-ID',
            'Folder-User',
            'Folder-Name',
            'From',
            'To',
            'Suspicious-Folders',
            'Date',
            'Message-ID',
            'Content-Type',
            'X-From',
            'X-To',
            'X-cc',
            'X-bcc',
            'X-Origin',
            'X-Folder',
            'Cc',
            'Attendees',
            'Bcc',
            'Time',
            'X-FileName',
            'Re',
            'Source',
            'POI-Present',
            'Low-Comm',
            'Contains-Reply-Forwards',
            'Sender-Type',
            'Unique-Mails-From-Sender',
        ],
        inplace=True,
        errors='ignore',
    )

    data['Subject'] = data['Subject'].fillna('no_subject').astype(str)
    data['Body'] = data['Body'].fillna('').astype(str)
    data['Text'] = data['Subject'] + ' ' + data['Body']

    text_series = data['Text'].astype(str)

    # engineering features into data - using vectorized operations for performance
    data['subject_len'] = data['Subject'].str.len()
    data['body_len'] = data['Body'].str.len()
    data['text_len'] = text_series.str.len()

    data['subject_word_count'] = data['Subject'].str.split().str.len()
    data['body_word_count'] = data['Body'].str.split().str.len()
    data['text_word_count'] = text_series.str.split().str.len()

    data['exclamation_count'] = text_series.str.count('!')
    data['question_count'] = text_series.str.count(r'\?')
    
    # Vectorized pattern matching using str.contains
    data['url_count'] = text_series.str.count(URL_PATTERN)
    data['digit_count'] = text_series.str.count(r'\d')
    
    # Uppercase ratio using vectorized operations
    uppercase_chars = text_series.str.findall(r'[A-Z]').str.len().fillna(0)
    text_len = text_series.str.len()
    data['uppercase_ratio'] = uppercase_chars / text_len.clip(lower=1)

    # Pattern checks using str.contains
    data['contains_reply'] = data['Subject'].str.contains(REPLY_PATTERN, case=False, na=False).astype(int)
    data['contains_forward'] = data['Subject'].str.contains(FORWARD_PATTERN, case=False, na=False).astype(int)
    data['has_html'] = text_series.str.contains(HTML_PATTERN, na=False).astype(int)
    
    # Mean word length using vectorized split
    data['mean_word_len'] = data['Body'].apply(_safe_mean_word_length)

    data.drop(columns=['Subject', 'Body'], inplace=True)

    return data
