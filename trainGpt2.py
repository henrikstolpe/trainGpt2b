from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json


#this is specific for GPT2

print("dataset formatting")

#original_dataset = load_dataset("samdotme/vader-speak")

# this function is used to output the right format for each row in the dataset
#def create_text_row(instruction, output):
#    text_row = f"""Question: {instruction}. Answer: {output}"""
#    return text_row

# iterate over all the rows and store the final format as a giant text file
#def save_file(output_file_path):
#    with open(output_file_path, "w") as output_file:
#        for item in original_dataset["train"]:
#            output_file.write(create_text_row(item["prompt"], item["response"]) + "\n")

# Provide the path where we want to save the formatted dataset
#save_file("./training_dataset.txt")

# We now load the formatted dataset from the text file
#dataset = load_dataset('text', data_files={'train': 'training_dataset.txt'})
dataset = load_dataset('text', data_files={'train': 'vader_luke_dialogue.txt'})

print("dataset loaded!")

print("starting tokenizer")

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Important for GPT-2 to get correct results.
tokenizer.pad_token = tokenizer.eos_token

# Worth uncommenting for understanding of the dataset structure.
print(dataset['train']['text'])

# Tokenize the data
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("loading model")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

print("setup training args")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

print("starting training")
# Train the model
trainer.train()
# Save the model
trainer.save_model("./my_gpt2")

from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("loading finetuned model")

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./my_gpt2')

print("loading tokenizer")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# We house the LLM query logic in a function so we can call it easily later.

def take_prompt_output_response(prompt):
 # Tokenize the input prompt
  input_ids = tokenizer.encode(prompt, return_tensors='pt')

  # Create attention mask (1 for real tokens, 0 for padding tokens)
  attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

  # Generate text
  output = model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_length=100,  # Adjust the max length to control the output length
      num_return_sequences=1,
      no_repeat_ngram_size=2,
      top_k=50,
      top_p=0.95,
      temperature=0.7,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id to eos_token_id
  )
  print("text generated")

# Decode and print the generated text
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  print(generated_text)

# Call the model with the prompt
print("ask a dataset specific question")
take_prompt_output_response("Do you ever wish you had a bigger boat or better equipment?.")
print("ask a generic question")
take_prompt_output_response("The earth is filled with wonder.")


