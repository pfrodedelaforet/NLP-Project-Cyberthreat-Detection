from transformers import  LongformerTokenizer, DataCollatorForLanguageModeling, Trainer, LongformerForMaskedLM, TrainingArguments
from tqdm import tqdm
from dataset_pytorch import Dataset_unlabelled_by_line
import torch
"""Script used to finetune the Longformer (without ProSeNet) on masked language modelling using the cybersecurity corpus."""
tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 512)

train_set = Dataset_unlabelled_by_line(filename = '../storage/block_emb/cybersec_train.txt')

test_set = Dataset_unlabelled_by_line(filename = '../storage/block_emb/cybersec_test.txt')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir = '../results/finetune',
    num_train_epochs = 1,#5
    per_device_train_batch_size = 2,#8
    gradient_accumulation_steps = 4,    
    per_device_eval_batch_size= 2,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    #load_best_model_at_end=True,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = True,
    logging_dir='../results/logging_finetune',
    #dataloader_num_workers = 0,
    run_name = 'longformer-classification-updated-rtx3090_paper_replication_2_warm', 
)

model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', attention_window=512, gradient_checkpointing = True)
model.to(torch.device('cuda:0'))
model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                               train_dataset=train_set, eval_dataset=test_set)

trainer.train()
trainer.save_model('pretrained_finetune')
trainer.evaluate()
