
import torch
import textwrap
from flask import Flask, jsonify ,request
from flask_restful import Resource, Api
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question3 = "How many parameters does BERT-large have?"
answer_text = "Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).[8] The disease was first identified in December 2019 in Wuhan, the capital of China's Hubei province, and has since spread globally, resulting in the ongoing 2019–20 coronavirus pandemic.[9][10] Common symptoms include fever, cough, and shortness of breath.[5] Other symptoms may include fatigue, muscle pain, diarrhea, sore throat, loss of smell, and abdominal pain.[5][11][12] The time from exposure to onset of symptoms is typically around five days, but may range from two to fourteen days.[5][13] While the majority of cases result in mild symptoms, some progress to viral pneumonia and multi-organ failure.[9][14] The virus is mainly spread between people during close contact,[a] often via small droplets produced during coughing,[b] sneezing, or talking.[6][15][17] While these droplets are produced when breathing out, they usually fall to the ground or onto surfaces rather than being infectious over large distances.[6][18][19] People may also become infected by touching a contaminated surface and then their face.[6][15] The virus can survive on surfaces for up to 72 hours.[20] It is most contagious during the first three days after onset of symptoms, although spread may be possible before symptoms appear and in later stages of the disease.[21] The standard method of diagnosis is by real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab.[22] Chest CT imaging may also be helpful for diagnosis in individuals where there is a high suspicion of infection based on symptoms and risk factors but is not recommended for routine screening"

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text,add_special_tokens=True)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # For each token and its id...
    
    # ======== Set Segment IDs ========


    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)
    

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return( answer )

app = Flask(__name__)
api = Api(app)
#answer_question(question3, answer_text)
class Quotes(Resource):
    def get(self, question): 
        args = request.args
        print("hudfbidhfgbuedhafdhufbdfhuidfhubdfuhdfiuhdfbdfiuhdfiuhdifubhdfibuh")
        print('question is ' + question)
        return jsonify({'ANSWER': answer_question(question, answer_text) }) 

api.add_resource(Quotes, '/<string:question>')

if __name__ == '__main__':
    app.run(debug=True)