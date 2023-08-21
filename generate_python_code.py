from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


checkpoint = 'Salesforce/codet5p-770M-py'
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float32,
                                              trust_remote_code=True).to(device)


def generate_python_code_based_on_user_input(input_value):
    input_value = input_value

    encoding = tokenizer(input_value, return_tensors='pt').to(device)
    encoding['decoder_input_ids'] = encoding['input_ids'].clone()

    outputs = model.generate(**encoding, max_length=750)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
