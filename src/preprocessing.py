import pandas as pd

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns= ['Mime-Version', 'Content-Transfer-Encoding', 'Mail-ID', 'Folder-User','Folder-Name','From', 
                        'To','Suspicious-Folders', 'Date', 'Message-ID','Content-Type','X-From',
                        'X-To', 'X-cc', 'X-bcc', 'X-Origin', 'X-Folder',  'Cc', 'Attendees',
                        'Bcc', 'Time', 'X-FileName', 'Re', 'Source', 'POI-Present', 'Suspicious-Folders', 'Low-Comm',
                        'Contains-Reply-Forwards', 'Sender-Type', 'Unique-Mails-From-Sender'], axis= 1, inplace= True)
    
    data['Subject']= data['Subject'].fillna('no_subject')
    data.drop(index= [259341, 387463, 147232, 147338], inplace= True)
    
    data['Text']= data['Subject'] + data['Body']
    data.drop(columns= ['Subject', 'Body'], inplace= True)

    return data