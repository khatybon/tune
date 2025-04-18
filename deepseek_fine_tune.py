import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

dataset = load_dataset("json", data_files="dataset_test.json")

if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 256  # Reduced max_length for memory efficiency

def tokenize_function(examples):
    inputs = [
        ", ".join(f"{key}: {value}" for key, value in eval(financial_data).items()) 
        if isinstance(financial_data, str) else 
        ", ".join(f"{key}: {value}" for key, value in financial_data.items())  
        for financial_data in examples["Financial_Data"]
    ]

    outputs = [", ".join(output) if isinstance(output, list) else str(output) for output in examples["Recommendations"]]

    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = tokenizer(
        outputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["Scenario_ID", "Industry", "Financial_Data", "Detected_Issues", "Best_Solutions", "Recommendations"]
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,  # Reduced from 16
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./deepseek_finetuned2",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,  # Reduce batch size to fit in memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # More steps to simulate larger batch size
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_checkpointing=True,  # Saves memory
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=default_data_collator
)

trainer.train()

model_save_path = "./deepseek_finetuned_final2"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

def generate_recommendations(financial_data):
    """ Test function to generate recommendations from input financial data """
    input_text = ", ".join(f"{key}: {value}" for key, value in financial_data.items())
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Test
sample_financial_data = {
    "Revenue": 343870582,
    "Revenue_Growth": -7.83,
    "Gross_Profit": 124929865.48,
    "Operating_Income": 71832133.24,
    "Net_Income": 39075447.42,
    "Total_Assets": 1365591325.601637,
    "Total_Liabilities": 685744893.15067,
    "Debt_to_Equity": 1.01,
    "Current_Ratio": 1.48,
    "Interest_Coverage": 10.79,
    "EBITDA": 80821743.48,
    "Free_Cash_Flow": 44212892.7,
    "ROE": 22.36,
    "Market_Cap": 3412099642.5575333
}

print("Generated Recommendations:", generate_recommendations(sample_financial_data))
