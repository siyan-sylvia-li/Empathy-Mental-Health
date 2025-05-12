from .roberta import RobertaForTokenClassification, RobertaModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch.nn.functional as F

from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable

from transformers import GPT2Model
from transformers import AutoModelWithLMHead, AutoTokenizer


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
	"roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
	"roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
	"roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
	"distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
	"roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
	"roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class Norm(nn.Module):
	def __init__(self, d_model, eps = 1e-6):
		super().__init__()
	
		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
		/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout = 0.1):
		super().__init__()
		
		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads
		
		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)
	
	def forward(self, q, k, v, mask=None):
		
		bs = q.size(0)
				
		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
			   
		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)

		scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
		
		concat = scores.transpose(1,2).contiguous()\
		.view(bs, -1, self.d_model)
		
		output = self.out(concat)
	
		return output
	
	def attention(self, q, k, v, d_k, mask=None, dropout=None):
	
		scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
		
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)
		
		scores = F.softmax(scores, dim=-1)
			
		if dropout is not None:
			scores = dropout(scores)
			
		output = torch.matmul(scores, v)
		
		return output


class SeekerEncoder(BertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.roberta = RobertaModel(config)
		self.init_weights()
	
	def get_input_embeddings(self):
		return self.roberta.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.roberta.embeddings.word_embeddings = value
	

class ResponderEncoder(BertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)
		self.roberta = RobertaModel(config)
		self.init_weights()
	
	def get_input_embeddings(self):
		return self.roberta.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.roberta.embeddings.word_embeddings = value

class BiEncoderAttentionWithRationaleClassification(nn.Module):

	def __init__(self, hidden_dropout_prob=0.2, rationale_num_labels=2, empathy_num_labels=3, hidden_size=768, attn_heads = 1):
		super().__init__()

		self.dropout = nn.Dropout(hidden_dropout_prob)
		self.rationale_classifier = nn.Linear(hidden_size, rationale_num_labels)
		self.attn = MultiHeadAttention(attn_heads, hidden_size)
		self.norm = Norm(hidden_size)
		self.rationale_num_labels = rationale_num_labels
		self.empathy_num_labels = empathy_num_labels
		self.empathy_classifier = RobertaClassificationHead(hidden_size = 768)

		self.apply(self._init_weights)

		self.seeker_encoder = RobertaModel.from_pretrained("roberta-base")
		self.responder_encoder = RobertaModel.from_pretrained("roberta-base")

		# Freeze seeker encoder
		for param in self.seeker_encoder.parameters():
			param.requires_grad = False

	
	def _init_weights(self, module):
		""" Initialize the weights """
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			initializer_range=0.02
			module.weight.data.normal_(mean=0.0, std=initializer_range)
		elif isinstance(module, BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


	# @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids_SP=None,
		input_ids_RP=None,
		attention_mask_SP=None,
		attention_mask_RP=None,
		token_type_ids_SP=None,
		token_type_ids_RP=None,
		position_ids_SP=None,
		position_ids_RP=None,
		head_mask_SP=None,
		head_mask_RP=None,
		inputs_embeds_SP=None,
		inputs_embeds_RP=None,
		empathy_labels=None,
		rationale_labels=None,
		lambda_EI=1,
		lambda_RE=0.1
	):
		# Get seeker post encodings
		outputs_SP = self.seeker_encoder(input_ids=input_ids_SP, attention_mask=attention_mask_SP)

		# Get response post encodings
		outputs_RP = self.responder_encoder(input_ids=input_ids_RP, attention_mask=attention_mask_RP)

		sequence_output_SP = outputs_SP[0]
		sequence_output_RP = outputs_RP[0]

		# Apply attention
		sequence_output_RP = sequence_output_RP + self.dropout(
			self.attn(sequence_output_RP, sequence_output_SP, sequence_output_SP)
		)

		# Get empathy classification logits
		logits_empathy = self.empathy_classifier(sequence_output_RP[:, 0, :])
		
		# Get rationale classification logits
		sequence_output = self.dropout(sequence_output_RP)
		logits_rationales = self.rationale_classifier(sequence_output)

		loss_rationales = 0.0
		loss_empathy = 0.0

		if rationale_labels is not None:
			loss_fct = CrossEntropyLoss()
			if attention_mask_RP is not None:
				active_loss = attention_mask_RP.view(-1) == 1
				active_logits = logits_rationales.view(-1, self.rationale_num_labels)
				active_labels = torch.where(
					active_loss, rationale_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale_labels)
				)
				loss_rationales = loss_fct(active_logits, active_labels)
			else:
				loss_rationales = loss_fct(logits_rationales.view(-1, self.rationale_num_labels), rationale_labels.view(-1))

		if empathy_labels is not None:
			loss_fct = CrossEntropyLoss()
			loss_empathy = loss_fct(logits_empathy.view(-1, self.empathy_num_labels), empathy_labels.view(-1))
			loss = lambda_EI * loss_empathy + lambda_RE * loss_rationales
			return loss, loss_empathy, loss_rationales, logits_empathy, logits_rationales

		return logits_empathy, logits_rationales



class RobertaClassificationHead(nn.Module):
	"""Head for sentence-level classification tasks."""

	def __init__(self, hidden_dropout_prob=0.1, hidden_size=768, empathy_num_labels=3):
		super().__init__()

		self.dense = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(hidden_dropout_prob)
		self.out_proj = nn.Linear(hidden_size, empathy_num_labels)

	def forward(self, features, **kwargs):
		x = features[:, :]  # take <s> token (equiv. to [CLS])
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x