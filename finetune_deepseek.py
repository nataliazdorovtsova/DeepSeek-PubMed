#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import sys
import urllib.request
import wandb
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    SchedulerType,
)
import matplotlib.pyplot as plt
import tqdm


# In[2]:


import torch


# In[ ]:


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)


# In[ ]:


# Read the wandb API key from file
with open("wandb_api_key.txt", "r") as f:
    api_key = f.read().strip()

# Log in
wandb.login(key=api_key)


# In[5]:


sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 3e-5, 1e-4]
        },
        'batch_size': {
            'values': [2, 4]
        },
        'warmup_steps': {
            'values': [10, 20]
        },
        'num_train_epochs': {
            'values': [3, 4]
        },
        'max_length': {
            'values': [1024, 2048]  # varying this to see if it's important to capture later details of papers, or if we can get away with less
        }
    }
}


# In[ ]:


# Custom Dataset that streams articles on the fly
class PMCDataset(Dataset):
    def __init__(self, file_list, base_url, tokenizer, max_length=512):
        self.file_paths = file_list
        self.base_url = base_url
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        url = self.base_url + path
        try:
            response = urllib.request.urlopen(url)
            article_bytes = response.read()
            article_text = article_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            article_text = ""
        tokenised = self.tokenizer(
            article_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        tokenised = {key: value.squeeze(0) for key, value in tokenised.items()}
        return tokenised

# Callback that collects loss values and plots them at the end of training.
class LossPlottingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, label="Training Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Across Steps")
        plt.legend()
        plt.grid(True)
        plt.show()


# In[ ]:


# Let's go

def main():
    # Initialise wandb without a fixed name.
    wandb.init(
        project="deepseek-finetuning",
        entity="nzdorovtsova",
        tags=["finetuning", "DeepSeek", "cosine", "experiment"],
        notes="Finetuning DeepSeek-R1 1.5B on PMC open access articles.",
        settings=wandb.Settings(init_timeout=180)
    )

    # Update run name dynamically based on the hyperparameters provided by the sweep.
    config = wandb.config
    wandb.run.name = f"lr{config.learning_rate}_epochs{config.num_train_epochs}_bs{config.batch_size}_maxlen{config.max_length}"
    
    # Retrieve hyperparameters directly from wandb.config (provided by the sweep agent)
    config = wandb.config
    num_train_epochs = config.num_train_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    warmup_steps = config.warmup_steps
    max_length = config.max_length
    
    # Configurable parameters for dataset
    FILE_LIST = "oa_file_list.txt"
    BASE_URL = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/"
    NUMBER_OF_ARTICLES = 10  # For proof-of-concept
    
    # Read and sample the file list.
    with open(FILE_LIST, "r", encoding="utf-8") as f:
        file_paths = [line.split('\t')[0] for line in f if line.strip()]
    if len(file_paths) > NUMBER_OF_ARTICLES:
        file_paths = random.sample(file_paths, NUMBER_OF_ARTICLES)
    
    # Load tokenizer and model from Hugging Face.
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create the streaming dataset.
    dataset = PMCDataset(file_paths, BASE_URL, tokenizer, max_length=max_length)
    
    # Define training arguments with cosine annealing and wandb integration.
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="no",
        fp16=True,  # Use mixed precision if supported
        lr_scheduler_type=SchedulerType.COSINE,
        warmup_steps=warmup_steps,
        report_to=["wandb"],
        disable_tqdm=False  # Ensure tqdm progress is visible in VSCode's terminal.
    )
    
    # Create the Trainer instance with our loss plotting callback.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[LossPlottingCallback()],
    )
    
    # Start fine-tuning.
    trainer.train()

# Main entry point - for now we are always running as a sweep agent.
if __name__ == "__main__":
    # Create a new sweep and immediately launch the sweep agent.
    sweep_id = wandb.sweep(sweep_config, project="deepseek-finetuning")
    print("Sweep ID:", sweep_id)
    wandb.agent(sweep_id, function=main)


# In[ ]:




