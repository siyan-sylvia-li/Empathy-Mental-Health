import codecs
import numpy as np
import pandas as pd
import re
import math
import random
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from models.models import BiEncoderAttentionWithRationaleClassification
from evaluation_utils import *

parser = argparse.ArgumentParser("BiEncoder")

parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--lambda_EI", default=0.5, type=float, help="lambda_identification")
parser.add_argument("--lambda_RE", default=0.5, type=float, help="lambda_rationale")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
parser.add_argument("--max_len", default=64, type=int, help="maximum sequence length")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=4, type=int, help="number of epochs")
parser.add_argument("--seed_val", default=12, type=int, help="Seed value")
parser.add_argument("--train_path", type=str, help="path to input training data")
parser.add_argument("--dev_path", type=str, help="path to input validation data")
parser.add_argument("--test_path", type=str, help="path to input test data")
parser.add_argument("--do_validation", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_model_path", type=str, help="path to save model")
parser.add_argument("--load_model_path", type=str, help="path to load pre-trained model")

args = parser.parse_args()

print("=====================Args====================")
print('lr = ', args.lr)
print('lambda_EI = ', args.lambda_EI)
print('lambda_RE = ', args.lambda_RE)
print('dropout = ', args.dropout)
print('max_len = ', args.max_len)
print('batch_size = ', args.batch_size)
print('epochs = ', args.epochs)
print('seed_val = ', args.seed_val)
print('train_path = ', args.train_path)
print('do_validation = ', args.do_validation)
print('do_test = ', args.do_test)
print("=============================================")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Set device to MPS if available
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f'Using device: {device}')

# Load data
df = pd.read_csv(args.train_path)

# Convert empathy labels to tensor
df['empathy_labels'] = df['level'].apply(lambda x: torch.tensor(x, dtype=torch.long))

# Robust conversion for rationale_labels
def parse_rationales(s):
	target_len = args.max_len
	if pd.isna(s) or s == '':
		return torch.tensor([0] * target_len, dtype=torch.long)
	s_clean = re.sub(r'[^0-9,]', '', str(s))
	try:
		vals = [int(i) for i in s_clean.split(',') if i != '']
		# Ensure length target_len: pad with zeros or truncate
		if len(vals) < target_len:
			vals = vals + [0] * (target_len - len(vals))
		elif len(vals) > target_len:
			vals = vals[:target_len]
		return torch.tensor(vals, dtype=torch.long)
	except Exception:
		return torch.tensor([0] * target_len, dtype=torch.long)

df['rationale_labels'] = df['rationales'].apply(parse_rationales)

# Tokenize input
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create datasets
input_ids_RP = tokenizer(df.response_post.tolist(), padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')['input_ids']
attention_masks_RP = tokenizer(df.response_post.tolist(), padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')['attention_mask']
input_ids_SP = tokenizer(df.seeker_post.tolist(), padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')['input_ids']
attention_masks_SP = tokenizer(df.seeker_post.tolist(), padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')['attention_mask']
labels = torch.stack(df['empathy_labels'].tolist())
rationales = torch.stack(df['rationale_labels'].tolist())

train_dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP, labels, rationales)
train_size = len(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

if args.do_test:
	if args.test_path:
		df_test = pd.read_csv(args.test_path, delimiter=',')
		df_test['rationale_labels'] = df_test['rationales'].apply(parse_rationales)
	else:
		print('No input test data specified.')
		print('Exiting...')
		exit(-1)

if args.do_validation:
	if args.dev_path:
		df_val = pd.read_csv(args.dev_path, delimiter=',')
		df_val['rationale_labels'] = df_val['rationales'].apply(parse_rationales)
	else:
		print('No input validation data specified.')
		print('Exiting...')
		exit(-1)

# Tokenize input
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(texts, max_length):
	return tokenizer(
		texts,
		padding='max_length',
		truncation=True,
		max_length=max_length,
		return_tensors='pt'
	)

# Tokenize training data
tokenizer_RP = tokenize_function(df.response_post.tolist(), args.max_len)
tokenizer_SP = tokenize_function(df.seeker_post.tolist(), args.max_len)

input_ids_RP = tokenizer_RP['input_ids']
attention_masks_RP = tokenizer_RP['attention_mask']
input_ids_SP = tokenizer_SP['input_ids']
attention_masks_SP = tokenizer_SP['attention_mask']

labels = torch.tensor(df.level.values.astype(int))
rationales = torch.stack(df.rationale_labels.values.tolist(), dim=0)

if args.do_validation:
	val_tokenizer_RP = tokenize_function(df_val.response_post.tolist(), args.max_len)
	val_tokenizer_SP = tokenize_function(df_val.seeker_post.tolist(), args.max_len)
	
	val_input_ids_RP = val_tokenizer_RP['input_ids']
	val_attention_masks_RP = val_tokenizer_RP['attention_mask']
	val_input_ids_SP = val_tokenizer_SP['input_ids']
	val_attention_masks_SP = val_tokenizer_SP['attention_mask']
	
	val_labels = torch.tensor(df_val.level.values.astype(int))
	val_rationales = torch.stack(df_val.rationale_labels.values.tolist(), dim=0)

if args.do_test:
	test_tokenizer_RP = tokenize_function(df_test.response_post.tolist(), args.max_len)
	test_tokenizer_SP = tokenize_function(df_test.seeker_post.tolist(), args.max_len)
	
	test_input_ids_RP = test_tokenizer_RP['input_ids']
	test_attention_masks_RP = test_tokenizer_RP['attention_mask']
	test_input_ids_SP = test_tokenizer_SP['input_ids']
	test_attention_masks_SP = test_tokenizer_SP['attention_mask']
	
	test_labels = torch.tensor(df_test.level.values.astype(int))
	test_rationales = torch.stack(df_test.rationale_labels.values.tolist(), dim=0)

# Load model
model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=args.dropout)
if args.load_model_path:
	print(f"Loading pre-trained model from {args.load_model_path}")
	model.load_state_dict(torch.load(args.load_model_path))
model = model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

# Create datasets
train_dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP, labels, rationales)
train_size = len(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

if args.do_validation:
	val_dataset = TensorDataset(val_input_ids_SP, val_attention_masks_SP, val_input_ids_RP, val_attention_masks_RP, val_labels, val_rationales)
	validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)

if args.do_test:
	test_dataset = TensorDataset(test_input_ids_SP, test_attention_masks_SP, test_input_ids_RP, test_attention_masks_RP, test_labels, test_rationales)
	test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

# Training schedule
total_steps = len(train_dataloader) * args.epochs
num_batch = len(train_dataloader)

print('total_steps =', total_steps)
print('num_batch =', num_batch)
print("=============================================")

scheduler = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=0,
	num_training_steps=total_steps
)

# Set random seeds
random.seed(args.seed_val)
np.random.seed(args.seed_val)
torch.manual_seed(args.seed_val)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(args.seed_val)

# Training loop
for epoch_i in range(0, args.epochs):
	total_train_loss = 0
	total_train_empathy_loss = 0
	total_train_rationale_loss = 0

	pbar = tqdm(total=num_batch, desc=f"training")

	model.train()

	for step, batch in enumerate(train_dataloader):
		b_input_ids_SP = batch[0].to(device)
		b_input_mask_SP = batch[1].to(device)
		b_input_ids_RP = batch[2].to(device)
		b_input_mask_RP = batch[3].to(device)
		b_labels = batch[4].to(device)
		b_rationales = batch[5].to(device)
		
		model.zero_grad()        

		loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
			input_ids_SP=b_input_ids_SP,
			input_ids_RP=b_input_ids_RP, 
			attention_mask_SP=b_input_mask_SP,
			attention_mask_RP=b_input_mask_RP, 
			empathy_labels=b_labels,
			rationale_labels=b_rationales,
			lambda_EI=args.lambda_EI,
			lambda_RE=args.lambda_RE
		)

		total_train_loss += loss.item()
		total_train_empathy_loss += loss_empathy.item()
		total_train_rationale_loss += loss_rationale.item()

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		optimizer.step()
		scheduler.step()

		pbar.set_postfix_str(
			f"total loss: {float(total_train_loss/(step+1)):.4f} epoch: {epoch_i}")
		pbar.update(1)

	pbar.close()
	
	# Save model if requested
	if args.save_model:
		torch.save(model.state_dict(), args.save_model_path)
		print(f"Model saved to {args.save_model_path}")

	# Validation
	if args.do_validation:
		model.eval()
		total_val_loss = 0
		total_val_empathy_loss = 0
		total_val_rationale_loss = 0
		
		for batch in validation_dataloader:
			b_input_ids_SP = batch[0].to(device)
			b_input_mask_SP = batch[1].to(device)
			b_input_ids_RP = batch[2].to(device)
			b_input_mask_RP = batch[3].to(device)
			b_labels = batch[4].to(device)
			b_rationales = batch[5].to(device)
			
			with torch.no_grad():
				loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
					input_ids_SP=b_input_ids_SP,
					input_ids_RP=b_input_ids_RP,
					attention_mask_SP=b_input_mask_SP,
					attention_mask_RP=b_input_mask_RP,
					empathy_labels=b_labels,
					rationale_labels=b_rationales,
					lambda_EI=args.lambda_EI,
					lambda_RE=args.lambda_RE
				)
				
				total_val_loss += loss.item()
				total_val_empathy_loss += loss_empathy.item()
				total_val_rationale_loss += loss_rationale.item()
		
		avg_val_loss = total_val_loss / len(validation_dataloader)
		print(f"Validation Loss: {avg_val_loss:.4f}")

	# Test
	if args.do_test:
		model.eval()
		total_test_loss = 0
		total_test_empathy_loss = 0
		total_test_rationale_loss = 0
		
		for batch in test_dataloader:
			b_input_ids_SP = batch[0].to(device)
			b_input_mask_SP = batch[1].to(device)
			b_input_ids_RP = batch[2].to(device)
			b_input_mask_RP = batch[3].to(device)
			b_labels = batch[4].to(device)
			b_rationales = batch[5].to(device)
			
			with torch.no_grad():
				loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
					input_ids_SP=b_input_ids_SP,
					input_ids_RP=b_input_ids_RP,
					attention_mask_SP=b_input_mask_SP,
					attention_mask_RP=b_input_mask_RP,
					empathy_labels=b_labels,
					rationale_labels=b_rationales,
					lambda_EI=args.lambda_EI,
					lambda_RE=args.lambda_RE
				)
				
				total_test_loss += loss.item()
				total_test_empathy_loss += loss_empathy.item()
				total_test_rationale_loss += loss_rationale.item()
		
		avg_test_loss = total_test_loss / len(test_dataloader)
		print(f"Test Loss: {avg_test_loss:.4f}")