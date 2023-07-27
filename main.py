from transformers import pipeline
import gradio

bert_NER = pipeline("ner",model="dslim/bert-base-NER")

def NER(input):
    output = bert_NER(input)
    return {"text":input, "entities":output}

gradio.close_all()
# demo = gradio.Interface(fn=summarize,inputs="text", outputs="text")
demo = gradio.Interface(fn=NER,
                        inputs=[gradio.Textbox(label="Text to find entities",lines=7)], 
                        outputs=[gradio.HighlightedText(label="Entities",lines=3)],
                        title="Name-Entity Recognition with dslim/bert-base-NER model")
demo.launch(share=True)