#!/usr/bin/env python
# coding: utf-8

# In[38]:

import json
import mlflow
import os
import argparse
import torch
import hf_olmo
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from transformers import set_seed

set_seed(123)



def main(args):
    # Your script logic goes here
    

    dataset = load_dataset('lmsys/lmsys-chat-1m')
    dataset = dataset.filter(lambda conv: conv['language'] == 'English', num_proc=4) # filtering all the other languages
    dict_cols = [key for key in dict(dataset['train'][0])]
    model_ckpt = "allenai/OLMo-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    special_tokens = ["<|im_start|>", "<|im_end|>"] # creating special tokens to append to the LLM
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        
    def format_chat(ex):
       
        chat = ex['conversation']
        
        formatted_chat = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False,
        )+ tokenizer.eos_token 
        
        tokenized_output = tokenizer(
                formatted_chat,
                add_special_tokens = False,
                padding="max_length",
                max_length=1024,
                truncation=True
        )
        
        return tokenized_output

    lmsys_tokenized = dataset.map(format_chat, num_proc=16).remove_columns(dict_cols)

    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    

    data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



    olmo_tokenized_split = lmsys_tokenized["train"].train_test_split(
        train_size=60000, test_size=10000)
    


    mlflow_tracking_path = os.path.join('../mlflow_results', args.mlflow_exp_name)

    
    
    if not os.path.exists(mlflow_tracking_path):
        os.mkdir(mlflow_tracking_path)
    


    OUTPUT_DIR = os.path.join(mlflow_tracking_path, 'output')
    LOG_DIR = os.path.join(mlflow_tracking_path, "logs")


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        warmup_steps=1,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        logging_steps=5,  # Log every 5 steps
        evaluation_strategy="epoch",
        lr_scheduler_type="linear",
        bf16=True,
        gradient_checkpointing=False,
        save_steps=1000,
        learning_rate=1e-5
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=olmo_tokenized_split["train"],
        eval_dataset=olmo_tokenized_split["test"],
        data_collator=data_collator,
    )
    


    mlflow.set_tracking_uri(mlflow_tracking_path)
    mlflow.set_experiment(args.mlflow_exp_id)
    with mlflow.start_run(log_system_metrics=True):
        mlflow.log_params(training_args.to_dict())
        trainer.evaluate() # eval before starting tuning
        trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLMO chat finetuning script")
    parser.add_argument("--mlflow_exp_name", type=str, default="olmo1b_chat", help="MLflow experiment name for chat finetuning", required=False)
    parser.add_argument("--mlflow_exp_id", type=str, default="701987021258002861", help="MLflow experiment ID for chat finetuning", required=False)
    args = parser.parse_args()

    main(args)
