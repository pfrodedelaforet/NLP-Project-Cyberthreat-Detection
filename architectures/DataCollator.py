import torch
class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        labels = torch.LongTensor([example[1] for example in examples])
        texts = [example[0] for example in examples]
        tokenizer_output = self.tokenizer(texts, truncation=True, padding='max_length', max_length = 2048)
        tokenizer_output['input_ids'] = torch.tensor(tokenizer_output['input_ids'])
        tokenizer_output['attention_mask'] = torch.tensor(tokenizer_output['attention_mask'])
        output_dict = dict(labels=labels, **tokenizer_output)
        return output_dict