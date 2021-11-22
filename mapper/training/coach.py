import os
import time

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils
from mapper.training.train_utils import convert_s_tensor_to_list

# Try to make the code work both on CPU/GPU
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = device
		self.opts.device = self.device

		# Initialize network
		self.net = StyleCLIPMapper(self.opts).to(self.device)

		self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

		# Initialize loss
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		if self.opts.clip_lambda > 0:
			self.clip_loss = clip_loss.CLIPLoss(opts)
		if self.opts.latent_l2_lambda > 0:
			self.latent_l2_loss = nn.MSELoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		descriptions = self.opts.description
		self.descriptions = descriptions.split(",")
		print(self.descriptions)
		with torch.no_grad():
			self.text_inputs = torch.cat([clip.tokenize(x) for x in self.descriptions]).to(self.device)
			self.text_embedding = self.clip_model.encode_text(self.text_inputs).to(self.device)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()

		total_time = 0.0

		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):

				with torch.no_grad():
					num_texts = len(self.text_inputs)
					num_samples = batch.size()[0]
					weights = torch.ones(size=(num_texts,)) * 1.0 / num_texts
					indices = torch.multinomial(weights, num_samples, replacement=True)
					text_batch = self.text_inputs[indices]
					text_embedding = self.text_embedding[indices]

					# `w` has shape = (batch_size, 18 (?), latent_dim)
					latent_dim = text_embedding.size()[-1]
					repeat = batch.size()[1]

					shape = (num_samples, 1, latent_dim)
					text_embedding = text_embedding.view(shape)
					shape = (num_samples, repeat, latent_dim)
					text_embedding = torch.broadcast_to(text_embedding, shape)

				s = time.time()

				self.optimizer.zero_grad()
				if self.opts.work_in_stylespace:
					w = convert_s_tensor_to_list(batch)
					w = [c.to(self.device) for c in w]
				else:
					w = batch
					w = w.to(self.device)
				with torch.no_grad():
					x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
				if self.opts.work_in_stylespace:

					# w_extended = w
					# w_extended = torch.cat([w, text_embedding], dim=-1)
					w_extended = w + text_embedding

					delta = self.net.mapper(w_extended)
					w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
					x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
				else:

					# w_extended = w
					# w_extended = torch.cat([w, text_embedding], dim=-1)
					w_extended = w + text_embedding

					w_hat = w + 0.1 * self.net.mapper(w_extended)
					x_hat, w_hat, _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				loss, loss_dict = self.calc_loss(w, x, w_hat, x_hat, text_batch)
				loss.backward()
				self.optimizer.step()

				e = time.time()
				total_time += (e - s)
				average_time_per_step = total_time / (self.global_step + 1)

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 1000 == 0):
					self.parse_and_log_images(x, x_hat, title='images_train')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')
					print(f"average time per step: {average_time_per_step}")

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if batch_idx > 200:
				break

			with torch.no_grad():
				num_texts = len(self.text_inputs)
				num_samples = batch.size()[0]
				weights = torch.ones(size=(num_texts,)) * 1.0 / num_texts
				indices = torch.multinomial(weights, num_samples, replacement=True)
				text_batch = self.text_inputs[indices]
				text_embedding = self.text_embedding[indices]

				# `w` has shape = (batch_size, 18 (?), latent_dim)
				latent_dim = text_embedding.size()[-1]
				repeat = batch.size()[1]

				shape = (num_samples, 1, latent_dim)
				text_embedding = text_embedding.view(shape)
				shape = (num_samples, repeat, latent_dim)
				text_embedding = torch.broadcast_to(text_embedding, shape)

			if self.opts.work_in_stylespace:
				w = convert_s_tensor_to_list(batch)
				w = [c.to(self.device) for c in w]
			else:
				w = batch
				w = w.to(self.device)

			with torch.no_grad():
				x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=self.opts.work_in_stylespace)
				if self.opts.work_in_stylespace:

					# w_extended = w
					# w_extended = torch.cat([w, text_embedding], dim=-1)
					w_extended = w + text_embedding

					delta = self.net.mapper(w_extended)
					w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
					x_hat, _, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1, input_is_stylespace=True)
				else:

					# w_extended = w
					# w_extended = torch.cat([w, text_embedding], dim=-1)
					w_extended = w + text_embedding

					w_hat = w + 0.1 * self.net.mapper(w_extended)
					x_hat, w_hat, _ = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				loss, cur_loss_dict = self.calc_loss(w, x, w_hat, x_hat, text_batch)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(x, x_hat, title='images_val', index=batch_idx)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.mapper.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.latents_train_path:
			train_latents = torch.load(self.opts.latents_train_path)
		else:
			train_latents_z = torch.randn(self.opts.train_dataset_size, 512).to(self.device)
			train_latents = []
			for b in range(self.opts.train_dataset_size // self.opts.batch_size):
				with torch.no_grad():
					_, train_latents_b, _ = self.net.decoder([train_latents_z[b: b + self.opts.batch_size]],
														  truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
					train_latents.append(train_latents_b)
			train_latents = torch.cat(train_latents)

		if self.opts.latents_test_path:
			test_latents = torch.load(self.opts.latents_test_path)
		else:
			test_latents_z = torch.randn(self.opts.train_dataset_size, 512).to(self.device)
			test_latents = []
			for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
				with torch.no_grad():
					_, test_latents_b, _ = self.net.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
													  truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
					test_latents.append(test_latents_b)
			test_latents = torch.cat(test_latents)

		if self.opts.work_in_stylespace:
			train_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in train_latents],
			                                                opts=self.opts)
			test_dataset_celeba = StyleSpaceLatentsDataset(latents=[l.cpu() for l in test_latents],
			                                     opts=self.opts)
		else:
			train_dataset_celeba = LatentsDataset(latents=train_latents.cpu(),
			                                      opts=self.opts)
			test_dataset_celeba = LatentsDataset(latents=test_latents.cpu(),
			                                      opts=self.opts)
		train_dataset = train_dataset_celeba
		test_dataset = test_dataset_celeba
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, w, x, w_hat, x_hat, text_batch):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement = self.id_loss(x_hat, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.clip_lambda > 0:
			### batched_text_inputs = torch.broadcast_to(self.text_inputs, (x_hat.size()[0], self.text_inputs.size()[1])).to(self.device)
			loss_clip = self.clip_loss(x_hat, text_batch).mean()
			loss_dict['loss_clip'] = float(loss_clip)
			loss += loss_clip * self.opts.clip_lambda
		if self.opts.latent_l2_lambda > 0:
			if self.opts.work_in_stylespace:
				loss_l2_latent = 0
				for c_hat, c in zip(w_hat, w):
					loss_l2_latent += self.latent_l2_loss(c_hat, c)
			else:
				loss_l2_latent = self.latent_l2_loss(w_hat, w)
			loss_dict['loss_l2_latent'] = float(loss_l2_latent)
			loss += loss_l2_latent * self.opts.latent_l2_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			#pass
			print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, x, x_hat, title, index=None):
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=self.opts.batch_size)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict