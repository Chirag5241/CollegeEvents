# more common imports
import pandas as pd
import numpy as np
from collections import Counter
import re

from PIL import Image
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

from importlib_metadata import os
# from nbclient import client

import boto3
import os

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

tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier", use_fast=False)
model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")

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


day_list = {"monday":0, "tuesday":1, "wednesday":2, "thursday":3, "friday":4, "saturday":5, "sunday":6}
month_list = {"jan":1, "feb":2, "march":3, "apr":4, "may":5, "june":6, "july":7, "august":8, "sept":9, "october":10, "nov":11, "december":12}
relative_date_list = {"today":0, "tomorrow": 1, "tonight": 0, "night":0}

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


test_data = {}
test_data['id'] = []
test_data['text'] = []
test_data['author'] = []


for folder_name in extract_array:
  cnt = 0
  id_extracted_list = []
  for dirname, dirs, files in os.walk('drive/MyDrive/Moment/data/'+folder_name):
      for filename in files:
          filename_without_extension, extension = os.path.splitext(filename)
          check_names = ["event", "announcement", "bullet", "meet", "opportuni", "intern"]                                 # Find channels with these keywords
          flag = False
          if extension == ".json":
              for name in check_names:
                  if name in filename:
                      flag = True
              if flag:
                  print(dirname)
                  # json.load(dirname+"\\"+)
                  print(filename.split("-"))
                  org = filename.split("-")[0]                                    # Get Organization name
                  with open(dirname+"/"+filename, encoding="utf8") as json_file:
                      data = json.load(json_file)
                      print(data['messages'][0]['id'])

                      name_disc = data['guild']['name']
                      for message in data['messages']:
                        # text = message['content']

                        if message['id']+'_disc' in id_extracted_list:      #if its alreaady extracted prevent repetiton
                          print(" continue here id:", message['id']+'_disc')
                          continue
                        id_extracted_list.append(message['id']+'_disc')

                        try:
                          date_time_stacks = get_time(message)
                        except:
                          print("date time EXCEPTION!!!!")
                          date_time_stacks = []
                          exception_count+=1
                          print(exception_count)
                          pass
                        if date_time_stacks != []:
                          # if len(message['content']) > 100:
                            # print(message)
                          print(message['content'])
                          content = re.sub(r'^https?:\/\/.*[\r\n]*', '', message['content'], flags=re.MULTILINE)
                          content = re.sub(r'^http?:\/\/.*[\r\n]*', '', content, flags=re.MULTILINE)
                          print(content)

                          test_data['id'].append(message['id'])
                          test_data['text'].append(content)
                          test_data['author'].append(folder_name)
                          
                            # break

                          # if message['id']+'_disc' in id_extracted_list:      #if its alreaady extracted prevent repetiton
                          #   print(" continue here id:", message['id']+'_disc')
                          #   continue
                          # id_extracted_list.append(message['id']+'_disc')