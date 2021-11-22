import os
from argparse import Namespace

import torchvision
import numpy as np
import clip
import torch
from torch.utils.data import DataLoader
import sys
import time

from tqdm import tqdm

from mapper.training.train_utils import convert_s_tensor_to_list

sys.path.append(".")
sys.path.append("..")

from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper

# Try to make the code work both on CPU/GPU
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"


clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def run(test_opts):
	out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = StyleCLIPMapper(opts)
	net.eval()
	net.to(device)

	test_latents = torch.load(opts.latents_test_path)
	if opts.work_in_stylespace:
		dataset = StyleSpaceLatentsDataset(latents=[l.cpu() for l in test_latents], opts=opts)
	else:
		dataset = LatentsDataset(latents=test_latents.cpu(), opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	description = opts.description

	global_i = 0
	global_time = []
	for input_batch in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			if opts.work_in_stylespace:
				input_cuda = convert_s_tensor_to_list(input_batch)
				input_cuda = [c.to(device) for c in input_cuda]
			else:
				input_cuda = input_batch
				input_cuda = input_cuda.to(device)

			tic = time.time()
			result_batch = run_on_batch(input_cuda, net, opts.couple_outputs, opts.work_in_stylespace, description)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			im_path = str(global_i).zfill(5)
			if test_opts.couple_outputs:
				couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
				torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
			else:
				torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
			torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, net, couple_outputs=False, stylespace=False, description=None):
	w = inputs

	with torch.no_grad():

		text_inputs = torch.cat([clip.tokenize(description)]).to(device)
		text_embedding = clip_model.encode_text(text_inputs).to(device)

		# `w` has shape = (batch_size, 18 (?), latent_dim)
		num_samples = w.size()[0]
		latent_dim = text_embedding.size()[-1]
		repeat = w.size()[1]

		shape = (num_samples, 1, latent_dim)
		text_embedding = text_embedding.view(shape)
		shape = (num_samples, repeat, latent_dim)
		text_embedding = torch.broadcast_to(text_embedding, shape)

		# w_extended = w

		text_weight = net.text_weight(torch.cat([w, text_embedding], dim=-1))
		w_extended = w + text_weight * text_embedding

		if stylespace:
			delta = net.mapper(w_extended)
			w_hat = [c + 0.1 * delta_c for (c, delta_c) in zip(w, delta)]
			x_hat, _, w_hat = net.decoder([w_hat], input_is_latent=True, return_latents=True,
			                                   randomize_noise=False, truncation=1, input_is_stylespace=True)
		else:
			w_hat = w + 0.1 * net.mapper(w_extended)
			x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True,
			                                   randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat)
		if couple_outputs:
			x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=stylespace)
			result_batch = (x_hat, w_hat, x)
	return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
