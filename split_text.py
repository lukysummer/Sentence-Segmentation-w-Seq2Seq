import numpy as np
import torch
from itertools import chain
import re, string, random
random.seed(827)

import nltk
from konlpy.tag import Mecab
import hangul_jamo

from .Seq2Seq import Seq2Seq
from .charLSTM_english import CharLevelLSTM as CharLevelLSTM_ENG


_device = "cuda" if torch.cuda.is_available() else "cpu"

######################## 1. CONSTRUCT INFERENCE MODEL #########################
def sample(model, lang, encoded_txt, space_id, tags=None):
    ''' PREDICTION FUNCTION '''
    model.eval()
    if lang == "KOR": # Attention Model
        idx = 0
        incomplete = True
    
        with torch.no_grad():
            # make predictions for every ~200 jamos (since model's trained with sequence length of 200 jamos)
            while incomplete:   
                if idx+200 <= len(encoded_txt):
                    try:
                        last_space_idx = [i for i, val in enumerate(encoded_txt[idx:idx+200]) if val==space_id][-1]
                    except:
                        last_space_idx = idx+200
                    inputs = torch.from_numpy(np.array([encoded_txt[idx:idx+last_space_idx+1]]))
                    targets = torch.zeros((1, last_space_idx+1)).type(torch.LongTensor)
                    
                    if idx+200 == len(encoded_txt) :
                        incomplete = False
                else:
                    inputs = torch.from_numpy(np.array([encoded_txt[idx:]])) 
                    targets = torch.zeros((1,len(encoded_txt)-idx+1)).type(torch.LongTensor)
                    incomplete = False

                pred, att_weights = model(inputs.to(_device), 
                                          targets.to(_device), 
                                          tf_ratio = 0)      # pred:   [1, n_output, n_jamos before last space] 
                pred_seq = pred.squeeze(0) if idx==0 else torch.cat((pred_seq, pred.squeeze(0)), dim=1) # pred_seq: [n_output, n_jamos]
    
                if incomplete:
                    idx += last_space_idx+1 

        # running on CPU: if 2 numbers are same, picks 1st one as max / GPU: picks 2nd one as max (following GPU rule)
        for i in range(pred_seq.size(1)):
            if pred_seq[0][i] == pred_seq[1][i]:
                pred_seq[0][i] = pred_seq[1][i]+100

        pred_seq = pred_seq.max(dim=0)[1] #[1] extracts index

    elif lang == "ENG": # simple bLSTM model
        pred_seq = [] 
        for i in range(len(encoded_txt)):
            # put through the model one jamo & one tag at a time
            input = np.array([[encoded_txt[i]]])
            input = torch.from_numpy(input).to(_device)
    
            tag = np.array([[tags[i]]])
            tag = torch.from_numpy(tag).to(_device)
    
            if i==0:
                h = model.init_h(input)

            h = tuple([each.data for each in h])
            pred, h = model(input, tag, h)
    
            p = torch.sigmoid(pred).data
            if(_device == "cuda"):
                p = p.cpu() 
            p = int(np.round(p.numpy()[0][0]))
            
            pred_seq.append(p)
    return pred_seq


def load_model(model_path, lang, use_phoneme=False):
    if lang == 'KOR': # Attention Model
        model = Seq2Seq(n_vocab = 123, #n_vocab, 
                        n_embed_text = 128, 
                        n_embed_dec = 2, 
                        n_hidden_enc = 128, 
                        n_hidden_dec = 128, 
                        n_layers = 1, 
                        n_output = 2, #CrossEntropyLoss
                        dropout = 0.5)   
    elif lang == 'ENG': # simple bLSTM model
        n_vocab = 81
        n_tags = 33
        model = CharLevelLSTM_ENG(n_vocab=n_vocab,  
                                  n_tags=n_tags,
                                  n_embed_text=128,
                                  n_hidden=128, 
                                  n_layers=2, 
                                  n_out=1)       
    model = model.to(_device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    #print("model loaded with", model_path)
    
    return model


def split(raw_text, model, lang, use_phoneme=False):
    #################### 1. PREPROCESS & ENCODE TEXT INPUT ####################
    raw_text = raw_text.replace("\u2028", "\n").replace('’', "'").replace('`', "'").replace('‘', "'").replace('“', '"').replace('”', '"')

    if lang == "KOR":               
        english_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                        'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']        
        korean_ja = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                     'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'] 
        korean_mo = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'] 
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']       
        others = [' ', '!', "'", ',', '.', '-', '?', '~', '"']
        
        vocab = ['NULL'] + english_char + korean_ja + korean_mo + numbers  + others
        vocab2idx = {ch:idx for idx, ch in enumerate(vocab)}
        idx2vocab = {idx:ch for idx, ch in enumerate(vocab)}
        space_id = vocab2idx[" "]
        
        # eliminate any sinlge 
        #raw_text = "".join([r for r in raw])
        
        raw = ""
        for r_i, r in enumerate(raw_text[:-1]):
            # if there is no space between eos puncs (.?!) and next word, insert space
            if (r in ".?!") and (raw_text[r_i+1] != " ") and (raw_text[r_i+1] not in ".?!"):
                raw += r+" "
            else:
                raw += r
        raw_text = raw + raw_text[-1]
        
        raw_text_split_by_eos_puncs = list(chain(*[rrr.split("\n") for rrr in list(chain(*[rr.split("!") for rr in list(chain(*[r.split(".") for r in raw_text.split("?")]))]))]))
        raw_text_split_by_eos_puncs = [r for r in raw_text_split_by_eos_puncs if r!=""]
        raw_text_split_by_eos_puncs = [r+" " for r in raw_text_split_by_eos_puncs]

        encoded_texts = [[vocab2idx[ch] for ch in list(split_jamos(seg)) if (ch not in ["?","!", "."]) and (ch in vocab)] for seg in raw_text_split_by_eos_puncs]
        orig_dicts = [{idx:ch for idx,ch in enumerate([idx2vocab[i] for i in text])} for text in encoded_texts]
        
    elif lang == 'ENG':
        tagger = Mecab()
        
        n_tags = 33
        tag2id = {'CC': 0, 'CD': 1, 'DT': 2, 'IN': 3, 'JJ': 4, 'JJR': 5, 'JJS': 6, 'MD': 7, 'NN': 8, 'NNS': 9, 'POS': 10, 'PRP': 11, 'PRP$': 12, 'RB': 13, 'RBR': 14, 'SC': 15, 'SF': 16, 'SPACE': 17, 'SSC': 18, 'SSO': 19, 'SY': 20, 'TO': 21, 'VB': 22, 'VBD': 23, 'VBG': 24, 'VBN': 25, 'VBP': 26, 'VBZ': 27, 'WDT': 28, 'WP': 29, 'WP$': 30, 'WRB': 31, 'etc': 32}
        char2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 
                   'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 
                   '!': 26, '"': 27, '#': 28, '$': 29, '%': 30, '&': 31, "'": 32, '(': 33, ')': 34, '*': 35, '+': 36, ',': 37, '-': 38, 
                   '.': 39, '/': 40, ':': 41, ';': 42, '<': 43, '=': 44, '>': 45, '?': 46, '@': 47, '[': 48, '\\': 49, ']': 50, '^': 51, 
                   '_': 52, '`': 53, '{': 54, '|': 55, '}': 56, '~': 57, ' ': 58, '\t': 59, '\n': 60, '\r': 61, '\x0b': 62, '\x0c': 63, 
                   '0': 64, '1': 65, '2': 66, '3': 67, '4': 68, '5': 69, '6': 70, '7': 71, '8': 72, '9': 73, 
                   '–': 74, 'ã': 75, 'á': 76, 'é': 77, 'ü': 78, 'î': 79, '…': 80}
        space_id = char2id[" "]
        
        raw_text = raw_text.replace('—', '— ') 
        raw_text_within_word_punc_deleted = "\n".join([" ".join(["".join([r[0]] + [ch for ch_i, ch in enumerate(r[1:]) if (ch != ".") or (re.sub(r'[^\w\s]','',r[ch_i+1:]) == "")]) for r in rn.split()]) for rn in raw_text.replace('—', '--').split("\n")])

        texts = raw_text_within_word_punc_deleted.lower().split("\n")
        orig_dict = {idx:ch for idx,ch in enumerate(raw_text_within_word_punc_deleted)} # orig_dict contains Capital information
        
        tokens = [re.findall(r"[\w']+|[^\s\w]", t) for t in texts] # Tokenize words & punctuations separately
        pairs = [[nltk.pos_tag([w])[0] if w not in string.punctuation else tagger.pos(w)[0] for w in t] for t in tokens] # Produce list of (word, tag) pairs #####   
        
        segment_tags = [[pair[1] for pair in t] for t in pairs]
        segment_words = [[pair[0] for pair in t] for t in pairs]
        
        tags = []
        encoded_text = []
        new_line_idxs = []
        new_line_i = 0
        for i, seg in enumerate(segment_words): # FOR EACH SEGMENT
            orig_words = texts[i].split()
            orig_words_i = 0    
            
            temp_tags = []
            temp_words = []
            tag_word = ""
            for j in range(len(seg)): # FOR EACH WORD IN THE SEGMENT
                tag_word += seg[j] 
                ####### WHEN PREPROCESSING, ONLY INCLUDE LETTERS IN VOCAB #########
                encoded_word = [char2id[ch] for ch in seg[j] if ch in char2id.keys()] 
               
                if tag_word == orig_words[orig_words_i]:
                    temp_words.extend(encoded_word + [space_id])
                    temp_tags.extend([segment_tags[i][j]]*len(encoded_word) + ["SPACE"])
                    orig_words_i += 1
                    tag_word = ""                 
                else:
                    temp_words.extend(encoded_word)
                    temp_tags.extend([segment_tags[i][j]]*len(encoded_word)) 
                    
            tags.append(temp_tags)
            encoded_text.append(temp_words)
            new_line_i += len(temp_words)
            new_line_idxs.append(new_line_i-1)

        tags = [tag2id[t] if t in tag2id.keys() else n_tags-1 for t in list(chain(*tags))]

        encoded_text = list(chain(*encoded_text))   
        encoded_text = [e for e in encoded_text if e is not None]
  
    
    ########################## 2. MAKE PREDICTIONS ############################
    if lang == "KOR":
        pred_seqs=[]
        for text in encoded_texts:
            pred_seqs.append(sample(model, lang, text, space_id))
        # extract positions where prediction is 1 & last char position (since must split @ end of segment)
        pred_pos_idxses = [[i for i in range(len(seq)-1) if seq[i]==1] + [len(encoded_texts[i])-1] for i, seq in enumerate(pred_seqs)]
        
        # if split @ jamo, not @ space (attention error):
        edited = []
        for pred_i, pred in enumerate(pred_pos_idxses):
            tmp = []
            for idx in pred:
                if orig_dicts[pred_i][idx]!=" ": # if not split @ space character
                    try: # if space exists before split jamo in the original segment, split there instead
                        prev_space_idx = [i for i, val in enumerate(encoded_texts[pred_i][:idx]) if val==space_id][-1]
                        tmp.append(prev_space_idx)
                    except IndexError: # if space doesn't exist before the split jamo, just delete that split
                        pass
                else:
                    tmp.append(idx)
            edited.append(tmp)
            
        pred_pos_idxses = edited        
        #print("NUMBER OF SPLITS:", sum([len(p) for p in pred_pos_idxses]))
        
    elif lang == "ENG":
        pred_seq = sample(model, lang, encoded_text, space_id, tags)

        pred_pos_idxs = [i for i in range(len(pred_seq)) if pred_seq[i]==1] + [len(encoded_text)-1]
        pred_pos_idxs = list(set(pred_pos_idxs + new_line_idxs)) 
        pred_pos_idxs.sort()


    #################### 3. DECODE PREDICTIONS INTO LETTERS ##################
    pred_segments = []    
    if lang == "KOR":
        for text_i, encoded_text in enumerate(encoded_texts):    
            composed=False
            pos_i = 0
            prev_i = 0
            for i in range(len(encoded_text)): 
                if ((pos_i < len(pred_pos_idxses[text_i])) and (i==pred_pos_idxses[text_i][pos_i])):
                    if prev_i == 0:
                        tmp = hangul_jamo.compose("".join([orig_dicts[text_i][ii] for ii in range(0, min(i+1, len(encoded_text)), 1)]))
                    else:
                        tmp = hangul_jamo.compose("".join([orig_dicts[text_i][ii] for ii in range(prev_i+1, min(i+1, len(encoded_text)), 1)]))

                    composed=True
                    pred_segments.append(tmp)
                    pos_i += 1
                    prev_i = i
            if not composed:
                pred_segments.append(hangul_jamo.compose("".join([orig_dicts[text_i][ii] for ii in range(len(encoded_text))])))
                  
    elif lang=="ENG":
        pos_i = 0
        prev_i = 0
        for i in range(len(encoded_text)): 
            if ((pos_i < len(pred_pos_idxs)) and (i==pred_pos_idxs[pos_i])):
                if prev_i == 0: # if 1st segment
                    if lang=="KOR":
                        pred_segments.append(hangul_jamo.compose("".join([orig_dict[ii] for ii in range(0, min(i+1, len(encoded_text)), 1)])))
                    elif lang=="ENG":
                        try:
                            pred_segments.append("".join([orig_dict[ii] for ii in range(0, min(i+1, len(encoded_text)), 1)]))
                        except:
                            pass
                else: # if not 1st segment
                    if lang=="KOR":
                        pred_segments.append(hangul_jamo.compose("".join([orig_dict[ii] for ii in range(prev_i+1, min(i+1, len(encoded_text)), 1)])))
                    elif lang=="ENG":
                        pred_segments.append("".join([orig_dict[ii] for ii in range(prev_i+1, min(i+1, len(encoded_text)-1), 1)])) 
                pos_i += 1
                prev_i = i


    ########## 4. (ENGLISH ONLY) RESTORE SPLITS AT "." AFTER ACRONYM ##########        
    if lang == "ENG":   
        # POLISH PREDICTIONS w.r.t. Syntax
        pred_segments = [seg.replace("--", "—").replace("\n", "") for seg in pred_segments]
        pred_segments = [seg for seg in pred_segments if (seg != " ") and (seg != "\n")] # delete any blank segments
        pred_segments = [seg[1:] if (len(seg)>1) and (seg[0]==" ") else seg for seg in pred_segments] # delete any space at the start of segment
        pred_segments = [seg for seg in pred_segments if seg not in string.punctuation] # exclude any single punctuation 
        
        acronyms = ['mrs.', 'ms.', 'mr.', 'dr.', 'drs.', 
                    'jr.', 'sr.', 'fr.', 'rev.', 'lt.', 'sgt.', 'ltd.',
                    'ft.', 'sq.', 'd.c.', 'maj.', 'gen.', 
                    'hon.', 'capt.', 'esq.', 'col.', 'st.', 'ave.', 'co.']
        final = []
        pp=""
        for p in pred_segments:
            pp += p
            if p.split()[-1].lower() not in acronyms:
                final.append(pp)
                pp = ""
    else:
        final=pred_segments
    
    final = [f for f in final if f!=" " and len(f.split())>0]
    # Put single-quotation segment at the end of the previous segment
    finaal=[]
    for ff in final:
        if ff.strip() == '"' or ff.strip() == "'":
            finaal[-1] += ff
        else:
            finaal.append(ff)
    final=finaal


    ####################### 5. PUT BACK OOV CHARACTERS ######################## 
    orig_tokens = raw_text.split()
    orig_tokens_no_punc = [re.sub(r'[^\w\s]','',t) for t in orig_tokens]
    pred_first_tokens_no_punc = [re.sub(r'[^\w\s]', '', p.split()[0]).lower() for p in final]
    pred_last_tokens_no_punc = [re.sub(r'[^\w\s]', '', p.split()[-1]).lower() for p in final]
    
    pred_i = 0
    orig_i = 0
    final_final = []
    passed = False
    for i, token in enumerate(orig_tokens_no_punc):
        token = token.lower() #if lang=="ENG" else token
        
        # if original token (with puncs deleted) is same as predicted segment's last word (with puncs deleted)  
        if split_jamos(token) == split_jamos(pred_last_tokens_no_punc[pred_i]):
            # if the segment's 1st and last words are NOT the same OR if the segment is one-letter
            if (pred_first_tokens_no_punc[pred_i] != pred_last_tokens_no_punc[pred_i]) or len(final[pred_i].split()) == 1:
                final_final.append(" ".join(orig_tokens[orig_i:i+1]))
                pred_i += 1
                orig_i = i+1
            # if the segment's 1st and last words are the same (trickier case)
            else:
                if not passed:
                    passed=True
                else:
                    final_final.append(" ".join(orig_tokens[orig_i:i+1]))
                    pred_i += 1
                    orig_i = i+1
                    passed=False
        
        
    ################## 6. BREAK LONG SEGMENTS BY PUNCTUATION ##################
    seg_puncs = [";", ",", '"', '“'] if lang == "KOR" else [";", ",", "—", ":"]
    for punc in seg_puncs:
        if punc == ";":
            long_n_chars = 0  # text must be split by ";"
        else:
            long_n_chars = 40 if lang == "KOR" else 60
        by_punc_n_chars = 5 if lang == "KOR" else 10
        final_final = split_by_punc(final_final, punc, long_n_chars, by_punc_n_chars)    

    final_final = [seg[1:] if (len(seg)>1) and (seg[0]==" ") else seg for seg in final_final] # delete any space at the start of segment


    ########## 7. COMBINE NUMBERS SPLIT BY ","(>1,000) & "."(decimal) #########
    pop_idxs = []
    for f_i, f in enumerate(final_final[:-1]):
        check_comma = (f[-1] == ",")  + (f[-2].isdigit()) + (final_final[f_i+1][0].isdigit()) if len(f)>=2 else 0
        check_dot = (f[-1] == ".")  + (f[-2].isdigit()) + (final_final[f_i+1][0].isdigit()) if len(f)>=2 else 0
        if max(check_comma, check_dot) == 3:
            pop_idxs.append(f_i+1)
            final_final[f_i] += final_final[f_i+1]
    
    final_final = [f for f_i, f in enumerate(final_final) if f_i not in pop_idxs]
    
    ### Final edit for splits before closing quotation mark or closing bracket
    fff = []
    for f in final_final:
        if '" ' in f:
            if f[:2]=='" ': # if '" ' is in the beginning of the segment
                fff[-1] += '"'
                fff.append(f[2:])
            else: # if '" ' is in the middle of the segment
                splits = f.split('" ')
                fff.append(splits[0] + '" ')
                fff.append(splits[1])
        
        elif f[:2]==') ': # if ') ' is in the beginning of the segment
                fff[-1] += ')'
                fff.append(f[2:])
        else:
            fff.append(f)
    final_final=fff

    return final_final


def split_jamos(word):
    return hangul_jamo.decompose(word)


def split_by_punc(segs, punc, long_n_chars, by_punc_n_chars):
    ''' BREAK LONG SEGMENTS BY PUNCTUATION
            :segs: (list of strings) list of split segments
            :punc: (string) a punc with which to separate segments
            :long_n_chars: (int) min # of letters in a segment to be considered to split by the punctuation
            :by_punc_n_chars: (int) if a split part of a segment is shorter than this number of letters, recombine.
    '''
    shortened_segs = [] 
    for i, seg in enumerate(segs):
            if len(seg) > long_n_chars:
                seg_part = seg[:-2] if seg[0]!='"' or punc == '“' else seg[1:-2]
                if punc in seg_part:
                    if punc == '"' or punc == '“': # if quotation
                        quote_i = seg.find('"') if punc == '"' else seg.find('“')
                        if quote_i > 5:
                            splits = [seg[:quote_i], seg[quote_i:]]
                            shortened_segs.extend(splits)
                        else:
                            shortened_segs.append(seg)
                    else: # if other punctuation than quotation
                        splits = seg.split(punc)
                        new_splits = [splits[0] + punc]
                        new_idx = 0
                        for sp_i, sp in enumerate(splits[1:]):
                            p = punc if sp_i+1 < len(splits) - 1 else ""
                            if len(sp) <= by_punc_n_chars:
                                new_splits[new_idx] += sp + p
                            else:
                                new_splits.append(sp+p)
                                new_idx += 1
                        
                        if len(new_splits[0]) <= by_punc_n_chars:
                            new_splits[1] = new_splits[0] + new_splits[1]
                            new_splits = new_splits[1:]
                        
                        shortened_segs.extend(new_splits)
                else:
                    shortened_segs.append(seg)
            else:
                shortened_segs.append(seg) 

    return shortened_segs
