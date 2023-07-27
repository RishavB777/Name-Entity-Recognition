from transformers import pipeline
import gradio

bert_NER = pipeline("ner",model="dslim/bert-base-NER")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def NER(input):
    output = merge_tokens(bert_NER(input))
    return {"text":input, "entities":output}


# demo = gradio.Interface(fn=summarize,inputs="text", outputs="text")
demo = gradio.Interface(fn=NER,
                        inputs=[gradio.Textbox(label="Text to find entities",lines=7)], 
                        outputs=[gradio.HighlightedText(label="Entities",lines=3)],
                        title="Name-Entity Recognition with dslim/bert-base-NER model")
demo.launch(share=True)
gradio.close_all()