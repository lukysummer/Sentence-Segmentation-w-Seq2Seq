import os, re, sys
import split_text

_model_path_KOR = 'models/model_formal_korean.pt'
_model_path_ENG = 'models/model_formal_english.pt'

# download model and store in model path
if not os.path.exists(_model_path_KOR):  
    raise RuntimeError(f'please download model to {_model_path_KOR}')
if not os.path.exists(_model_path_ENG):
    raise RuntimeError(f'please download model to {_model_path_ENG}')

# input arguments:
book_title = str(sys.argv[1])      # tile of the book to split into segments
print_result = bool((sys.argv[2])) # whether to print result in the window
save_result = bool((sys.argv[3]))  # whether to save result in a text file

class TextSplitter():

    def configure_lang(self, input_text):
        if re.sub('[^A-Za-z]+', '', input_text) == '': # if text contains ONLY korean letters (+ numbers + punctuations)
            lang = "KOR"
        else: # if text includes some alphabets
            if re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]+', '', input_text) != '': # if text contains some korean letters
                lang = "KOR"
            else:
                lang = "ENG"
                
        return lang, input_text
                
    def split(self, book_title):
        book_dir = "books/" + book_title + ".txt"
        with open(book_dir, "r") as f:
            input_text = f.read().strip()

        lang, input_text = self.configure_lang(input_text)
        model = split_text.load_model(_model_path_KOR, lang) if lang=="KOR" else split_text.load_model(_model_path_ENG, lang)
        pred_segments = split_text.split(input_text, model, lang)
        if print_result:
            print()
            for i, p in enumerate(pred_segments):
                print(i, ":", p)
        
        if save_result:
            with open("done_books/" + book_title + "_split.txt", "w") as f:
                for line_i, line in enumerate(pred_segments):
                    clean_text = line.replace("  ", " ").strip()
                    clean_text = clean_text[1:] if clean_text[0]==" " else clean_text
                    f.write(clean_text)
                    f.write("\n")

        return pred_segments

text_splitter = TextSplitter()
segments = text_splitter.split(book_title)
