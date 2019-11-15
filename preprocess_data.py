import os
import numpy as np
import pickle
import string
from tts_text_util.process_text_input import TextToInputSequence
from konlpy.tag import Mecab

tagger = Mecab()


######################## 1. READ IN TRANSCRIPTION FILES #######################
########### & SPLIT INTO TRAIN, VALID, TEST TEXTS FOR EACH SPEAKER ############
voicepath = "scripts"

train_ratio, valid_ratio = 0.8, 0.1
''' Test text & speaker ids must be a dict, because testing is done separately for each speaker. '''
speaker_dict = {}
train_texts, valid_texts, test_texts = [], [], []
train_speaker_ids, valid_speaker_ids, test_speaker_ids = [], [], []

idx = 0
for actor in os.listdir(voicepath):
    try:
        # 1. Read Text
        scriptpath = voicepath + "/" + actor + "/metadata.data"   
        with open(scriptpath) as f:
            # For Each Speaker:
            text = f.read().strip().split("\n")
            train_cutoff = int(len(text) * train_ratio)
            valid_cutoff = int(len(text) * (train_ratio+valid_ratio))
            
            train_texts.extend(text[:train_cutoff])
            valid_texts.extend(text[train_cutoff:valid_cutoff])
            test_texts.extend(text[valid_cutoff:])
            
            # 2. Read Speaker Index   
            speaker_dict[idx] = str(actor)
            train_speaker_ids.extend([idx] * train_cutoff)
            valid_speaker_ids.extend([idx] * (valid_cutoff-train_cutoff))
            test_speaker_ids.extend([idx] * (len(text)-valid_cutoff))
            idx += 1

            for t in text:
                if len(t.split("|"))<2:
                    print(str(actor)+":", t)
    except:
        pass

n_speaker = len(speaker_dict)
assert len(train_speaker_ids)==len(train_texts)
assert len(valid_speaker_ids)==len(valid_texts)
assert len(test_speaker_ids) == len(test_texts)


####################### 2. EXTRACT TEXT SCRIPTS ONLY ##########################
def extract_text(texts, speaker_ids, test=False):
    scripts = [s.split("|")[1] for s in texts]
    if not test:
        scripts = [s if (s[-1] not in string.punctuation) else s[:-1] for s in scripts]        
            
    assert len(speaker_ids)==len(scripts)

    return scripts, speaker_ids


train_texts, train_speaker_ids = extract_text(train_texts, train_speaker_ids)
valid_texts, valid_speaker_ids = extract_text(valid_texts, valid_speaker_ids)
test_texts, test_speaker_ids = extract_text(test_texts, test_speaker_ids)

n_segments = len(train_texts)


################### 4. ENCODE TEXT WITH PHONEME/JAMO INDEX ####################
use_phoneme = False   # Use Jamo 
insert_dot = False    # Don't insert dot at the end of each segment not ending in punctuations

text2seq = TextToInputSequence(use_phoneme=use_phoneme, kor_only=True)
n_vocab = text2seq.get_vocab_size()

train_text_path = 'encoded_scripts_train.txt'
train_speaker_path = 'speaker_ids_train.txt'
train_tag_path = 'tags_train.txt'

valid_text_path = 'encoded_scripts_valid.txt'
valid_speaker_path = 'speaker_ids_valid.txt'
valid_tag_path = 'tags_valid.txt'

test_text_path = 'encoded_scripts_test.txt'
test_speaker_path = 'speaker_ids_test.txt'
test_tag_path = 'tags_test.txt'

comma_id = text2seq.get_input_sequence(text=",", target_language='KOR', use_phoneme=use_phoneme, insert_dot=insert_dot)[0][0]
space_id = text2seq.get_input_sequence(text="가 나", target_language='KOR', use_phoneme=use_phoneme, insert_dot=insert_dot)[0][2]


def load_file(path):
    with open (path, 'rb') as fp:
        return pickle.load(fp)

def encode_text(texts, speaker_ids, text_path, speaker_path, tag_path):
    print("Encoding Text...")
        
    ############ 1. TAGS  (Segment 단위) ################
    segment_tags = [[pair[1] for pair in tagger.pos(t)[:]] for t in texts]
    segment_words = [[pair[0] for pair in tagger.pos(t)[:]] for t in texts]
    tags = []
    encoded_texts = []
    for i, seg in enumerate(segment_words): # FOR EACH SEGMENT
        orig_words = texts[i].split()
        orig_words_i = 0
            
        temp_tags = []
        temp_words = []
        tag_word = ""
        for j in range(len(seg)): # FOR EACH WORD IN THE SEGMENT
            tag_word += seg[j] 
            encoded_word = text2seq.get_input_sequence(text = seg[j],
                                                       target_language='KOR', 
                                                       use_phoneme=False,
                                                       insert_dot=False)[0] #+ [space_id]
            if tag_word == orig_words[orig_words_i]:
                temp_words.extend(encoded_word + [space_id])
                temp_tags.extend([segment_tags[i][j]]*len(encoded_word) + ["SPACE"])
                orig_words_i += 1
                tag_word = ""
            else:
                temp_words.extend(encoded_word)
                temp_tags.extend([segment_tags[i][j]]*len(encoded_word)) 
        tags.append(temp_tags)
        encoded_texts.append(temp_words)

    
    ############ 2. SPEAKER IDS  (Segment 단위) ################
    speaker_ids = [[speaker_ids[ii]]*len(encoded_texts[ii]) for ii in range(len(encoded_texts))]
    assert len(speaker_ids)==len(encoded_texts)  # SEGMENT 단위

    with open(text_path, 'wb') as fp:
        pickle.dump(encoded_texts, fp)
    with open(speaker_path, 'wb') as fp:
        pickle.dump(speaker_ids, fp)
    with open(tag_path, 'wb') as fp:
        pickle.dump(tags, fp)
        
    return encoded_texts, speaker_ids, tags


if os.path.exists(train_text_path):
    train_encoded_texts, train_speaker_ids, train_tags = load_file(train_text_path), load_file(train_speaker_path), load_file(train_tag_path)
    valid_encoded_texts, valid_speaker_ids, valid_tags = load_file(valid_text_path), load_file(valid_speaker_path), load_file(valid_tag_path)
    test_encoded_texts, test_speaker_ids, test_tags = load_file(test_text_path), load_file(test_speaker_path), load_file(test_tag_path)

else:
    train_encoded_texts, train_speaker_ids, train_tags = encode_text(train_texts, train_speaker_ids, train_text_path, train_speaker_path, train_tag_path)
    valid_encoded_texts, valid_speaker_ids, valid_tags = encode_text(valid_texts, valid_speaker_ids, valid_text_path, valid_speaker_path, valid_tag_path)
    test_encoded_texts, test_speaker_ids, test_tags = encode_text(test_texts, test_speaker_ids, test_text_path, test_speaker_path, test_tag_path)



############################# 5. CONSTRUCT X & Y ##############################
def create_x_s_y(encoded_texts, speaker_ids, tags):
    x, s, t, y = [], [], [], []
    for i in range(len(encoded_texts)):
        segment = encoded_texts[i] # do NOT remove punctuation at the end of each segment 
        speaker = speaker_ids[i]
        tag = tags[i]
    
        x.extend(segment)                    
        s.extend(speaker)
        t.extend(tag)
        y.extend([0]*(len(segment)-1) + [1]) 
        
    return np.array(x), np.array(s), t, np.array(y)
    

train_x, train_s, train_t, train_y = create_x_s_y(train_encoded_texts, train_speaker_ids, train_tags)
valid_x, valid_s, valid_t, valid_y = create_x_s_y(valid_encoded_texts, valid_speaker_ids, valid_tags)
test_x, test_s, test_t, test_y = create_x_s_y(test_encoded_texts, test_speaker_ids, test_tags)


all_tags = train_t + valid_t + test_t
all_tags = sorted(set(all_tags))
n_tags = len(all_tags) + 1
print()
print("Number of Tags:", n_tags)

tag2id = {tag:i for i, tag in enumerate(all_tags)}
tag2id["etc"] = n_tags - 1
train_t = np.array([tag2id[t] for t in train_t])
valid_t = np.array([tag2id[t] for t in valid_t])
test_t = np.array([tag2id[t] for t in test_t])


assert len(train_x)==len(train_y)
assert len(train_s)==len(train_x)
avg_n_jamos_per_segment = len(train_x)/n_segments

print("Number of Speakers:", n_speaker)
print("Number of Unique Phonemes/Jamos:  ", n_vocab)
print("Length of Training Data (in Segments): ", n_segments)
print("Length of Training Data (in Jamos)   : ", len(train_x))
print("Average Number of Jamos / Segment:   %.1f"%avg_n_jamos_per_segment)

 

################# 6. DEFINE BATCHING FUNCTION FOR TRAINING ####################

def get_batches(data_x, data_y, batch_size, seq_len, overlap, data_s=None, data_t=None):
    assert len(data_x) == len(data_y)
    n_batches = int(np.floor(len(data_x)/(batch_size*seq_len))) 
    print("n_batches:", n_batches)
    
    if data_s is not None:
        assert len(data_s) == len(data_x)
        data_s = data_s[:n_batches * batch_size * seq_len]
        data_s = data_s.reshape(batch_size, -1)
        
        assert len(data_t) == len(data_x)
        data_t = data_t[:n_batches * batch_size * seq_len]
        data_t = data_t.reshape(batch_size, -1)
        
    data_x = data_x[:n_batches * batch_size * seq_len]  # truncate data
    data_x = data_x.reshape(batch_size, -1) #[batch_size, n_batches*seq_len]
    data_y = data_y[:n_batches * batch_size * seq_len]
    data_y = data_y.reshape(batch_size, -1)

    for i in range(0, data_x.shape[1], overlap):
        try:
            if data_s is None:
                if (i+seq_len) <= data_x.shape[1]:
                    x = data_x[:, i:i+seq_len]
                    y = data_y[:, i:i+seq_len]
                    yield x, y
                else:   
                    pass
            else:
                if (i+seq_len) <= data_x.shape[1]:
                    x = data_x[:, i:i+seq_len]
                    s = data_s[:, i:i+seq_len]
                    t = data_t[:, i:i+seq_len]
                    y = data_y[:, i:i+seq_len]
                    yield x, s, t, y
                else:   
                    pass
        except: 
            pass
    
    
# example usage
batches = get_batches(test_x, test_y, batch_size=64, seq_len=100, overlap=50, data_t=test_t, data_s=test_s)
x,s,t,y = next(batches)