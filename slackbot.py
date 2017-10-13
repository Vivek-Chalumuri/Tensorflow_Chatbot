import os
import time
from slackclient import SlackClient
import tensorflow as tf
import numpy as np
import random
from datasets.cornell_corpus import data
import data_utils
import seq2seq_model

metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')
xseq_len = 25
yseq_len = 25
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
model = seq2seq_model.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/cornell_corpus/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

sess = model.restore_last_session()
# starterbot's ID as an environment variable
BOT_ID = "U7HTQ122H"

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "do"

# instantiate Slack & Twilio clients
slack_client = SlackClient('xoxb-255942036085-VoVSvzOkFrTMwujitLPVCnAL')

GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up",)

GREETING_RESPONSES = ["'sup bro", "hey", "*nods*", "hey you get my snap?"]

def check_for_greeting(sentence):
    """If any of the words in the user's input was a greeting, return a greeting response"""
    res = ''
    for word in sentence.split():
        if word.lower() in GREETING_KEYWORDS:
            res = random.choice(GREETING_RESPONSES)
            break
    return res


def handle_command(command, channel):
    
    res = check_for_greeting(command)
    
    if res == '':

        try:
           res = model.get_response(command, metadata, sess)
           words = res.split()
           if "unk" in words:
               words.remove('unk')
           res = " ".join(words)
        except ValueError:
           res = "Hmm"

    response = res

    if command.startswith(EXAMPLE_COMMAND):
        response = "Sure...write some more code then I can do that!"
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)


def parse_slack_output(slack_rtm_output):
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None


if __name__ == "__main__":
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
