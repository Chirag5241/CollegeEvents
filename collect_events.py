from PIL import Image
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

from importlib_metadata import os
# from nbclient import client

import boto3
import os

import pandas as pd

import json
import datetime
import re

from transformers import BertTokenizer, AutoTokenizer, BertForTokenClassification
import torch
from collections import Counter

from flair.data import Sentence
from flair.models import SequenceTagger

from neo4j import GraphDatabase

data_folder_name = '/home/ec2-user/d_scrape/dataUCSD/'

neo4j_uri = "neo4j+s://32c386b6.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "lXx1rWQyLKFNkRk3YbZrs0fNf8s5ujqBmA3HC5edcFk"

extract_array = os.listdir(data_folder_name)

def get_pred_type(prediction):

    if prediction == "O" or prediction == "[PAD]" or prediction == "[SEP]":
        return prediction
    else:
        return prediction.split("-")[1]

def get_vote_type(votes):
    # Since Python 3.7, Counter maintains insertion order.
    # Since we want to preserve the first label in case of ties, we need to reverse the votes,
    # as we previously recorded them backwards.
    votes = [get_pred_type(vote) for vote in reversed(votes)]
    majority = Counter(votes).most_common(1)
    majority_label = majority[0][0]

    return majority_label

def merge_tokens(bpe_text, bpe_predictions, id2label, tokenizer):
    """
    BPEs are merged into single tokens in this step, where corresponding predictions get aggregated
    into a single token by virtue of majority voting.
    Even breaks (e.g., something like "me ##ssa ##ge | B-DATE, O, I-DURATION") will be decided by the first tag result,
    in this case "DATE" because of the tag of "me". If there is no B-tag in the current instance at all,
    the first token still decides. Note that there are no ambiguities about the B/I distinction here, since we only
    look at multi-BPE tokens, and not at tags spanning multiple *full-word* tokens.
    TODO: Note that this function gets rid of the B/I distinction for downstream tasks as well currently!
      This can be changed by not abstracting the vote to the type only, and still carrying the B-/I- prefix with it.
    :param bpe_text:
    :param bpe_predictions:
    :param id2label: Turning predicted ids back to the actual labels
    :param tokenizer: Tokenizer required to translate token ids back to the words themselves.
    :return: List of tuples containing (token, type_label) pairs.
    """
    merged_tokens = []
    prev_multi_instance = False
    current_multi_vote = []
    current_multi_token = ""
    # Iterate in reverse to immediately see when we deal with a multi-BPE instance and start voting
    for token_id, pred_id, in zip(reversed(bpe_text), reversed(bpe_predictions)):
      # print(token_id.numpy())
      token = tokenizer.ids_to_tokens[int(token_id)]

      # print(pred_id)
      pred = id2label[int(pred_id)]

      # Skip special tokens
      if token in ("[PAD]", "[CLS]", "[SEP]"):
          continue

      # Instance for multi-BPE token
      if token.startswith("##"):
          current_multi_token = f"{token[2:]}{current_multi_token}"
          current_multi_vote.append(pred)
      else:
          # Need to merge votes
          if current_multi_token:
              current_multi_token = f"{token}{current_multi_token}"
              current_multi_vote.append(pred)
              merged_tokens.append((current_multi_token, get_vote_type(current_multi_vote)))
              current_multi_token = ""
              current_multi_vote = []
          # Previous token was single word anyways
          else:
              merged_tokens.append((token, get_pred_type(pred)))

    # Bring back into right order for later processing
    merged_tokens.reverse()
    return merged_tokens

def pred_time(input_text):
  processed_text = tokenizer(input_text, return_tensors="pt")
  result = model(**processed_text)
  classification= result[0]
  labeled_list = merge_tokens(processed_text['input_ids'][0], torch.max(classification[0],dim=1)[1], id2label, tokenizer)
  final_list = []
  for word in labeled_list:
    if word[1] != 'O':
      final_list.append(word)
  return final_list

def get_loc(sent):
  if sent == "":
    return []
  sentence = Sentence(sent)
  tagger.predict(sentence)

  location_stack = []

  # print the sentence with all annotations
  # print(sentence)

  sentence_lowercase = sent.lower()

  # print('The following NER tags are found:')

  # iterate over entities and print each
  for entity in sentence.get_spans('ner'):
    label = entity.get_label("ner").value
    if label == 'ORG' or label == 'LOC':
      # print(entity.text)
      if "club" not in entity.text.lower():
        location_stack.append(entity.text)

  loc_index = -1
  if location_stack != []:
    # print("Earliest loc: ",len(sent.split(location_stack[0])[0]))
    loc_index = len(sent.split(location_stack[0])[0])


  if 'zoom' in sentence_lowercase:    ### tag it accordding to its location
    a = re.search(r'\b(zoom)\b', sentence_lowercase)
    # print(a.start())
    if a.start() < loc_index or loc_index == -1:
      location_stack.append('Zoom')

  if 'info' in sentence_lowercase:
    location_stack.append('Zoom')

  return location_stack

def lfnc(test_date, weekday_idx): return test_date + \
          datetime.timedelta(days=(weekday_idx - test_date.weekday() + 7) % 7)

def lfnc2(test_date, relative_day): return test_date + datetime.timedelta(days=relative_day)

def lfnc3(test_date, month, day): return test_date.replace(day=day, month=month)


def get_time(message):
  out_list = []

  text = message['content']

  out_list = pred_time(text)
  
  if out_list == []:
    return []
  # print(out_list)

  time_stack = []
  date_stack = []
  time_of_day = "pm"

  date = datetime.datetime.strptime(message['timestamp'][:10], "%Y-%m-%d")

  i=0 
  while i < len(out_list):
    # print(out_list[i])

    if out_list[i][1] == 'TIME':
      flag = 1
      if 'pm' in out_list[i][0].lower() or 'am' in out_list[i][0].lower():  #check am or pm
        time_of_day = out_list[i][0][-2:]
        flag = 0

      if out_list[i][0].isnumeric() and i+2<len(out_list) and int(out_list[i][0]) <= 12: # if its formatted as x:yz
        if out_list[i+1][1] == 'TIME' and out_list[i+1][0].strip() == ":":  # if its formatted as x:yz
          time_stack.append(out_list[i][0]+out_list[i+1][0]+out_list[i+2][0][:2])
          i = i+2
          flag = 0
        else:
          time_stack.append(out_list[i][0]) # if its just x
          flag = 0

      elif out_list[i][0].lower().replace('pm','').replace('am','').isnumeric() and int(out_list[i][0].lower().replace('pm','').replace('am','')) <= 12:  # if its just x or x pm/x am
        # print("HEERERER",out_list[i])
        time_stack.append(out_list[i][0].lower().replace('pm','').replace('am',''))
        flag = 0
      
      if flag:
        res = -1
        for relative_day in relative_date_list:
          if relative_day in out_list[i][0].lower():
            res = lfnc2(date, relative_date_list[relative_day])
            break
        if res != -1:
          date_stack.append(str(res.strftime("%Y/%m/%d")))

    if out_list[i][1] == 'DATE' or out_list[i][1] == 'SET':    # if it has a date tag
      res = -1
      flag = 1

      possible_num = out_list[i][0].replace("th","").replace("rd","").replace("nd","").replace("st","")   ## this change was made for insta, reflect on discord
      if possible_num.isnumeric() and i+1 < len(out_list):  # ex: 5 jan
        for month in month_list:
          if month in out_list[i+1][0].lower() and int(possible_num) < 32:
            res = date.replace(day=int(possible_num), month=month_list[month])#lfnc3(date, month_list[month], int(out_list[i][0]))
            flag = 0
            i=i+1
            break
        if flag and int(possible_num) < 32:
          res = date.replace(day=int(possible_num))
          flag = 0

      if flag:
        for month in month_list:
          if month in out_list[i][0].lower() and i+1 < len(out_list):  #ex: jan 5
            possible_num = out_list[i+1][0].replace("th","").replace("rd","").replace("nd","").replace("st","")
            if possible_num.isnumeric() and int(possible_num) < 32:
              res = date.replace(day=int(possible_num), month=month_list[month])
              flag = 0
              i=i+1
            break

      # if out_list[i][0].isnumeric() and i+1 < len(out_list):  # ex: 5 jan
      #   for month in month_list:
      #     if month in out_list[i+1][0].lower() and int(out_list[i][0]) < 32:
      #       res = date.replace(day=int(out_list[i][0]), month=month_list[month])#lfnc3(date, month_list[month], int(out_list[i][0]))
      #       flag = 0
      #       i=i+1
      #       break
      #   if flag and int(out_list[i][0]) < 32:
      #     res = date.replace(day=int(out_list[i][0]))
      #     flag = 0

      if flag:
        for month in month_list:
          if month in out_list[i][0].lower() and i+1 < len(out_list):  #ex: jan 5
            possible_num = out_list[i+1][0].replace("th","").replace("rd","").replace("nd","").replace("st","")
            if possible_num.isnumeric() and int(possible_num) < 32:
              res = date.replace(day=int(possible_num), month=month_list[month])
              flag = 0
              i=i+1
            break

      if flag:
        for day in day_list:
          if day in out_list[i][0].lower():
            res = lfnc(date, day_list[day])
            flag = 0
            break 

      if flag:
        for relative_day in relative_date_list:
          if relative_day in out_list[i][0].lower():
            # print("HERE",out_list[i][0].lower())
            res = lfnc2(date, relative_date_list[relative_day])
            flag = 0
            break

      if res != -1:
        # print("Date: " + str(res)[:10])
        date_stack.append(str(res.strftime("%Y/%m/%d")))

    i+=1

  # print(text)
  # print(out_list)

  if time_stack != [] and date_stack != []:
    # print(text)
    # print(out_list)
    # print(time_of_day)
    # print(time_stack)
    # print(date_stack)
    return {"time_of_day": time_of_day, "time_stack": time_stack, "date_stack":date_stack}

  return []

def img_url_to_s3(profile_or_not, loc_name, url):
  loc_name_no_space = ''.join(('-' if ch in ' ' else ch) for ch in loc_name)

  if profile_or_not:  # if its using the default profile image
    # url = "https://test-bucket-chirag5241.s3.us-west-1.amazonaws.com/"+loc_name_no_space
    url = "https://moment-events.s3.us-east-2.amazonaws.com/"+loc_name_no_space
    try:
      resp = requests.get(url, stream=True).raw   # if this works, image exists
    except:
      print("\nNo default image!!!\n")
      return -1
    # return "https://test-bucket-chirag5241.s3.us-west-1.amazonaws.com/"+loc_name_no_space
    return "https://moment-events.s3.us-east-2.amazonaws.com/"+loc_name_no_space

  resp = requests.get(url, stream=True).raw
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  if image is None:
      print("\nNO IMAGE HERE! ####\n")
      return -1
  
  # print(image.shape[0])
  if image.shape[0] < 150 or image.shape[1] < 150:    # If the image is small, resize it
      dim = (image.shape[0]*4,image.shape[1]*4)       # new dimensions
      resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)   # resize image
      resized_blur = cv2.GaussianBlur(resized,(0,0),cv2.BORDER_DEFAULT)   # blur to remove from image
      image_sharp = cv2.addWeighted(resized, 1.5, resized_blur, -0.6, 0, resized_blur)
      kernel = np.array([[0, -1, 0],
                          [-1, 5,-1],
                          [0, -1, 0]])
      image = cv2.filter2D(src=image_sharp, ddepth=-1, kernel=kernel)
      # image = cv2.resize(image, (128,128), interpolation = cv2.INTER_CUBIC)

  # RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # plt.figure()
  # plt.imshow(RGB_im)

  # print("https://test-bucket-chirag5241.s3.us-west-1.amazonaws.com/"+loc_name_no_space)
  print("https://moment-events.s3.us-east-2.amazonaws.com/"+loc_name_no_space)


  ########

  image_string = cv2.imencode('.jpg', image)[1].tobytes()

  upload_file_key =  loc_name_no_space #'test-images/'+str(file)

  s3.Object(upload_file_bucket, upload_file_key).put(Body=image_string,ContentType='image/PNG')
  print("\nNEW Image uploaded to S3!!\n")

  # return "https://test-bucket-chirag5241.s3.us-west-1.amazonaws.com/"+loc_name_no_space
  return "https://moment-events.s3.us-east-2.amazonaws.com/"+loc_name_no_space
  

tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier", use_fast=False)
model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")

id2label = {v: k for k, v in model.config.label2id.items()}

print("id2label: ",id2label)

tagger = SequenceTagger.load("/home/ec2-user/.flair/models/ner-english-large/07301f59bb8cb113803be316267f06ddf9243cdbba92a4c8067ef92442d2c574.554244d3476d97501a766a98078421817b14654496b86f2f7bd139dc502a4f29")

# tagger = SequenceTagger.load("flair/ner-english-large")

print("Downloading complete")

day_list = {"monday":0, "tuesday":1, "wednesday":2, "thursday":3, "friday":4, "saturday":5, "sunday":6}
month_list = {"jan":1, "feb":2, "march":3, "apr":4, "may":5, "june":6, "july":7, "august":8, "sept":9, "october":10, "nov":11, "december":12}
relative_date_list = {"today":0, "tomorrow": 1, "tonight": 0, "night":0}


## AWS

access_key = 'AKIAR6GV237C6LRS6K7D' #'AKIA2IIOOLB6IZ4NQOWM'
secret_access_key = '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' #'YuCZO2+yId3Hj4yBwUXkuIxUiP12100pIH6V6TyW'  #change when putting in py file

# Creating Session With Boto3.
session = boto3.Session(
aws_access_key_id=access_key,
aws_secret_access_key=secret_access_key
)

#Creating S3 Resource From the Session.
s3 = session.resource('s3')

upload_file_bucket =  'moment-events' #test-bucket-chirag5241' #moment-events.s3.us-east-2

print("Extract ARRAY!!",extract_array)


df = pd.DataFrame(columns=["ID","CreatorID","Image", "Name", "startingTime","Location","Description","Invitation","Visibility"])
df_links = pd.DataFrame(columns=["ID","ID_connect","link"])
df_user = pd.DataFrame(columns=["ID","Name","Image"])

user_list = []

data_base_connection = GraphDatabase.driver(uri = neo4j_uri, auth=(neo4j_user, neo4j_password))

def push_to_neo4j(session,event_list):
  # from neo4j import GraphDatabase

  # transaction_list = transaction_DataFrame.values.tolist()

  # transaction_execution_commands = []

  # for i in transaction_list:
      
      # neo4j_create_statemenet = "create (t:Transaction {transaction_id:" + str(i[0]) +", vendor_number:  " + str(i[1]) +", transaction_amount: " + str(i[2]) +", transaction_type: '" + str(i[3]) + "'})"

  # data_base_connection = GraphDatabase.driver(uri = "neo4j+s://bff6d1fc.databases.neo4j.io", auth=("neo4j", "57yuB7Ts5UL1Ajbggx9kVLoovrIiHwAI0NZ1Veu2_I0"))
  # session = data_base_connection.session() 

  university_id = 'univ_UCSD'

  for event in event_list:
    print("\n Event pushing:",str(event[0]))
    # descp = ''.join( (ch if ch != '*' else "\n") for ch in str(color_quad[1]) ) 
    print(str(event[1]))
    print(str(event[2]))
    print(str(event[3]))
    print(str(event[4]))
    print(str(event[5]))
    print(str(event[6]))
    print(str(event[7]))
    print(str(event[8]))
    session.run('''MERGE (n:Event {ID: $ID})
        ON CREATE SET
          n.CreatorID = $CreatorID,
          n.Description	 = $Description,
          n.ID = $ID,
          n.Image = $Image,
          n.Invitation = $Invitation,
          n.Location = $Location, 
          n.Name = $Name,
          n.Visibility = $Visibility,
          n.startingTime = $startingTime,
          n.time_created = datetime()
        ON MATCH SET
          n.CreatorID = $CreatorID,
          n.Description	 = $Description,
          n.Image = $Image,
          n.Invitation = $Invitation,
          n.Location = $Location, 
          n.Name = $Name,
          n.Visibility = $Visibility,
          n.startingTime = $startingTime,
          n.time_created = datetime()''', ID=str(event[0]), CreatorID = str(event[1]),Image = str(event[2]), 
            Name = str(event[3]), startingTime = str(event[4]),Location = str(event[5]), Description = str(event[6]), Invitation = str(event[7]),   
            Visibility = str(event[8]), )


    print("\n Event Univ Join:",str(event[0]))
    session.run('''MATCH (u:University {ID: $Univ_ID}),(n:Event {ID: $ID})
                  MERGE (u)-[r:event_univ]->(n)''', ID=str(event[0]), Univ_ID = university_id)

    print("\n Event Organization Join:",str(event[0]))
    session.run(''' MATCH (u:Organization {ID: $CreatorID}),(n:Event {CreatorID: $CreatorID})
                    MERGE (u)-[r:Created]->(n)''', CreatorID = str(event[1]))
    
    # break

def push_to_neo4j_creator(session, event_list):

  university_id = 'univ_UCSD'

  for event in event_list:
    print("\nCreator ID: ",str(event[0]))
    # descp = ''.join( (ch if ch != '*' else "\n") for ch in str(color_quad[1]) ) 
    print(str(event[1]))
    print(str(event[2]), "\n")
    # print(str(event[3]))
    session.run('''MERGE (n:Organization {ID: $ID})
        ON CREATE SET
          n.ID = $ID,
          n.Image = $Image,
          n.Name = $Name,
          n.time_created = datetime()
        ON MATCH SET
          n.Image = $Image,
          n.Name = $Name,
          n.time_created = datetime()''', ID=str(event[0]), Name = str(event[1]), Image = str(event[2]))

    print("\n Univ organization join:",str(event[0]))
    session.run('''MATCH (u:University {ID: $Univ_ID}),(n:Organization {ID: $ID})
                  MERGE (n)-[r:at_univ]->(u)''', ID=str(event[0]), Univ_ID = university_id)
    
    # break


print(extract_array)

# exception_count = 0
# for folder_name in extract_array:
#   cnt = 0
#   id_extracted_list = []

#   data_base_connection = GraphDatabase.driver(uri = neo4j_uri, auth=(neo4j_user, neo4j_password))

#   for dirname, dirs, files in os.walk(data_folder_name+folder_name):

#     for file_id, filename in enumerate(files):

#       if file_id % 10 == 0:
#         print("#########SESSION")
#         session = data_base_connection.session()

#       # print(filename)
#       filename_without_extension, extension = os.path.splitext(filename)
#       check_names = ["event", "announcement", "bullet", "meet","opportuni", "intern","week","social"]                                # Find channels with these keywords
#       flag = False
#       if extension == ".json":
#         for name in check_names:
#             if name in filename:
#                 flag = True
#         if flag:
#           print(dirname)
#           # json.load(dirname+"\\"+)
#           print(filename.split("-"))
#           org = filename.split("-")[0]                                    # Get Organization name
#           with open(dirname+"/"+filename, encoding="utf8") as json_file:
#             data = json.load(json_file)
#             print(data['messages'][0]['id'])

#             default_image = data['guild']['iconUrl']          # set default image in case no image found

#             creator_id = data['guild']['name']+"_"+data['guild']['id']

#             default_image = img_url_to_s3(False, "images/discord/"+creator_id+"/server_img_"+data['guild']['name']+"_disc"+".jpg", default_image)

#             name_disc = data['guild']['name']
#             for message in data['messages']:
#               # text = message['content']

#               if message['id']+'_disc' in id_extracted_list:      #if its alreaady extracted prevent repetiton
#                 print(" continue here id:", message['id']+'_disc')
#                 continue
#               id_extracted_list.append(message['id']+'_disc')

#               try:
#                 date_time_stacks = get_time(message)
#               except:
#                 print("date time EXCEPTION!!!!")
#                 date_time_stacks = []
#                 exception_count+=1
#                 print(exception_count)
#                 pass
#               if date_time_stacks != []:
#                 loc_stack = get_loc(message['content']) 
#                 urls = re.findall(r'(https?://\S+)', message['content'])
#                 if loc_stack != [] or urls != []:
#                   # Date * Location * ID * Photo * Name * Time * Invitation Website/Link * Description Visibility (public/private)

#                   # Esxtract image from text
#                   image_url = default_image # set image 
#                   if message['attachments'] != []:
#                     for image in message['attachments']:    ## if there are attached images
#                       if 'png' in image['url'] or 'jpg' in image['url']:
#                         image_url = image['url']
#                         image_url = img_url_to_s3(False, "images/discord/"+creator_id+"/"+message['id']+'_disc'+".jpg", image_url)
#                         if image_url == -1:   # in the case that no image exists
#                           print("IMAGE URL ISSUE!!!")
#                           image_url = default_image
#                         break

#                   if image_url==default_image:  # if there was no attached image
#                     image_url = img_url_to_s3(True, "images/discord/"+creator_id+"/server_img_"+data['guild']['name']+"_disc"+".jpg", image_url)

#                   if image_url == -1:   # in the unexpected case that no image exists for both message and default
#                     print("BIG EXCEPTION FOR IMAGE!!!")
#                     image_url = default_image


#                   # ID=str(event[0]), CreatorID = str(event[1]),Image = str(event[2]), 
#                   #  Name = str(event[3]), startingTime = str(event[4]),Location = str(event[5]), Description = str(event[6]), Invitation = str(event[7]),   
#                   # Visibility = str(event[8]), 

#                   temp_dict = {}
#                   temp_dict["ID"] = message['id']+'_disc'
#                   temp_dict["CreatorID"] = creator_id+"_disc"
#                   temp_dict["Image"] = image_url
#                   temp_dict["Name"] = name_disc
#                   date_time_val = str(date_time_stacks['date_stack'][0]+" "+date_time_stacks['time_stack'][0].replace('pm','').replace('am','')+" "+date_time_stacks['time_of_day'].upper())
#                   if ":" not in date_time_val:
#                     date_time_val = date_time_val[:-3]+":00"+date_time_val[-3:]
#                   temp_dict["startingTime"] = date_time_val
#                   if loc_stack == []:
#                     temp_dict["Location"] = urls[0]
#                   else:
#                     temp_dict["Location"]=  loc_stack[0]
#                   temp_dict["Description"]= message['content'].replace('@','')
#                   temp_dict["Invitation"] = ""
#                   temp_dict["Visibility"] = "Public"

#                   print(cnt)
#                   cnt+=1
#                   # print(temp_dict["ID"])

#                   if temp_dict["CreatorID"] not in user_list:
#                     user_list.append(temp_dict["CreatorID"])
#                     temp_user_dict = {}
#                     temp_user_dict['ID'] = temp_dict["CreatorID"]
#                     temp_user_dict['Name'] = data['guild']['name']
#                     temp_user_dict['Image'] = default_image 

#                     creator_list = pd.DataFrame([temp_user_dict]).values.tolist()

#                     #push creator to neo4j
#                     push_to_neo4j_creator(session, creator_list)

#                     df_user = df_user.append(pd.DataFrame([temp_user_dict]),  ignore_index=False, verify_integrity=False, sort=None)
                  
#                   # temp_dict["Website/Link"] = urls
#                   for i,url in enumerate(urls):
#                     temp_link_dict = {}
#                     temp_link_dict['ID'] = message['id']+'_disc'+str(i)
#                     temp_link_dict['ID_connect'] = message['id']+'_disc'
#                     temp_link_dict['link'] = url
#                     df_links = df_links.append(pd.DataFrame([temp_link_dict]),  ignore_index=False, verify_integrity=False, sort=None)

#                   print(temp_dict)

#                   df = df.append(pd.DataFrame([temp_dict]),  ignore_index=True)

#                   # print(df)

#                   print("ID: ", message['id']+'_disc')
#                   print("CreatorID: ", temp_dict["CreatorID"])
#                   print("Image: ",image_url)
#                   print("Name: ", name_disc)
#                   print("startingTime: ",date_time_val)
#                   # print("Time: ")
#                   if loc_stack == []:
#                     print("Location: ", urls[0])
#                   else:
#                     print("Location: ", loc_stack[0])
                  
#                   print("Description: ", message['content'].replace('@',''))
#                   print("Website/Link: ", urls)

#                   event_list = pd.DataFrame([temp_dict]).values.tolist()

#                   print(event_list)

#                   push_to_neo4j(session, event_list)
                  



  # df.to_csv("drive/MyDrive/Social_Calendar/neo4j_data/discord_aws.csv",index=False)
  # df_links.to_csv("drive/MyDrive/Social_Calendar/neo4j_data/discord_aws_links.csv",index=False)
  # df_user.to_csv("drive/MyDrive/Social_Calendar/neo4j_data/discord_aws_users.csv",index=False)   
  # print("File Saved") 