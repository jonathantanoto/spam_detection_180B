import requests
import shutil
import os
import re
import datetime as dt
import sys
import pandas as pd
import zipfile as zf

print('Downloading Enron Dataset...')

base_url = 'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/'
files = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]

if not os.path.exists("../data/raw_enron_placeholder"):
    os.mkdir("../data/raw_enron_placeholder")

print()
for entry in files:
    print("Downloading folder: " + entry + "...")

    # Download current enron folder
    url = base_url + entry + ".tar.gz"
    r = requests.get(url)
    path = "../data/raw_enron_placeholder/" + entry + ".tar.gz"

    with open(path, 'wb') as f:
        f.write(r.content)
    print('-------Downloaded to: ' + path)

    # Unpack to /raw_enron_placeholder/enron1, etc
    print("Unpacking " + entry)
    shutil.unpack_archive(path, "../data/raw_enron_placeholder/")
    print("-------Folder unpacked to: raw_enron_placeholder/" + entry)
    
    print()

print("All downloads completed.")

print()

print("Initiating Data Processing")
      
# Data Processing
# Traverse messages in separate files, then parse and add to csv

dataset = []

print("Processing directories...")
# Traverse enron1,2,etc
# Each contains ham/spam folder
# Each has txt "message" files 
for folder in files:
    print("----Inside " + str(folder) + "...")
    ham_path = "../data/raw_enron_placeholder/" + folder + "/ham"
    spam_path = "../data/raw_enron_placeholder/" + folder + "/spam"

    # Process non-spams
    for entry in os.scandir(ham_path):
        file = open(entry, encoding="latin_1")
        content = file.read().split("\n", 1)
        
        sub = content[0].replace("Subject: ", "")
        message = content[1].replace('\n', ' ')
        
        pattern = r"\d+\.(\d+-\d+-\d+)"
        date = dt.datetime.strptime(re.search(pattern, str(entry)).group(1), '%Y-%m-%d')
        file.close()
        dataset.append([sub, message, "ham", date])

    # Process spams
    for entry in os.scandir(spam_path):
        file = open(entry, encoding="latin_1")
        content = file.read().split("\n", 1)
        
        sub = content[0].replace("Subject: ", "")
        message = content[1].replace('\n', ' ')
        
        pattern = r"\d+\.(\d+-\d+-\d+)"
        date = dt.datetime.strptime(re.search(pattern, str(entry)).group(1), '%Y-%m-%d')
        file.close()
        dataset.append([sub, message, "spam", date])

    print("-------" + str(folder)+" transformed to list")
    
    print()

print("Transforming list to DataFrame...")
mails = pd.DataFrame(dataset, columns=[
                     "Subject", "Message", "Spam/Ham", "Date"])

# Transform to csv
print("Saving as csv and zip...")
with zf.ZipFile('../data/enron_spam_data.zip', 'w') as enron_zip:
    enron_zip.writestr('../data/enron_spam_data.csv', mails.to_csv(index_label = "Message ID"), compress_type=zf.ZIP_DEFLATED)
print("-------saved as '../data/enron_spam_data.zip'")

print()

# Final count
print("\nData processed and saved to file.\nMails contained in data:")
print("\nTotal:\t" + str(mails.shape[0]))
print(mails["Spam/Ham"].value_counts())