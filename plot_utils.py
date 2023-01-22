import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_alignment (projs, d_output=10, epochs=None, out_dir='.', title=''):

	n_layers = len(projs) + 1
	n_snapshots = len(projs[0])
	if epochs is None:
		epochs = np.arange(n_snapshots)
	n_skip = epochs[1]-epochs[0]

	# ALIGNMENT
	kwargs=dict(vmin=0, vmax=1, aspect='equal') # cmap="bwr", vmin=-1, 
	cols=max(n_layers-1, 2)
	fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
	# plt.subplots_adjust(wspace=0.4)
	plt.subplots_adjust(hspace=0.3)
	def plot_frame (frame):
	    plt.cla()
	    fig.suptitle(title+f" -- epoch {frame*n_skip}")
	    # plot alignment of intermediate layers
	    for l, proj in enumerate(projs):
	        ax = axs[l]
	        ax.set_xlabel(r"$m$")
	        ax.set_ylabel(r"$n$")
	        ax.set_title(rf"$|V^n_{l+2}\cdot U^m_{l+1}$|")
	        im = ax.imshow(np.abs(proj[frame, :d_output+2, :d_output+2]), **kwargs)#; plt.colorbar(im, ax=ax)

	plot_frame(len(epochs)-1)
	fig.savefig(f'{out_dir}/alignment.png', bbox_inches="tight")

	duration = 10 # in s
	frames = np.linspace(0,len(epochs)-1,51).astype(int)
	dt = duration*1000./len(frames) # in ms
	ani = FuncAnimation(fig, plot_frame,
	                    interval=dt,
	                    frames=frames,
	                    blit=False)
	ani.save(f'{out_dir}/alignment.gif')

	# fig, axs_ = plt.subplots(1, 3, figsize=(14, 4))
	# axs = axs_.ravel()
	# # plt.subplots_adjust(wspace=0.4)
	# plt.subplots_adjust(hspace=0.3)
	# fig.suptitle(title)
	# ax = axs[0]
	# ax.set_ylim([0,1.1])
	# ax.set_xlabel("epoch")
	# ax.set_ylabel(r"$|V^n_3\cdot U^m_2|$")
	# dims = V3U2.shape
	# for i in range(d_output+1): # range(dims[1]):
	#     for j in range(d_output+1): #range(dims[2]):
	#         c = "C0" if i == j else "C1"
	#         ax.plot(epochs, np.abs(V3U2[:, i, j]), c=c)
	# ax = axs[1]
	# ax.set_ylim([0,1.1])
	# ax.set_xlabel("epoch")
	# ax.set_ylabel(r"$|V^n_2\cdot U^m_1|$")
	# dims = V2U1.shape
	# for i in range(d_output+1): # range(dims[1]):
	#     for j in range(d_output+1): #range(dims[2]):
	#         c = "C0" if i == j else "C1"
	#         ax.plot(epochs, np.abs(V2U1[:, i, j]), c=c)
	# # ax = axs[2]
	# # ax.set_ylim([0,1.1])
	# # ax.set_xlabel("epoch")
	# # ax.set_ylabel(r"$|V^n_1\cdot \tilde{V}^m|$")
	# # dims = V1Vw.shape
	# # for i in range(d_output+1): # range(dims[1]):
	# #     for j in range(d_output+1): #range(dims[2]):
	# #         c = "C0" if i == j else "C1"
	# #         ax.plot(epochs, np.abs(V1Vw[:, i, j]), c=c)
	# fig.savefig(f'{out_dir}/alignment_vs_epoch.png', bbox_inches="tight")


def plot_singular_values (Ss, epochs=None, out_dir='.', title=''):

	n_layers = len(Ss)
	n_snapshots = len(Ss[0])
	if epochs is None:
		epochs = np.arange(n_snapshots)

	# PARTICIPATION RATIO AND LARGEST SINGULAR VALUE
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(title)
	ax.set_xlabel('epoch')
	ax.set_ylabel('participation ratio / rank')
	ax.grid()
	PR = lambda S: np.array([np.sum(s)**2/np.sum(s**2)/len(s) for s in S])
	for l, S in enumerate(Ss):
	    ax.plot(epochs, PR(S), label=f"{l+1}")
	ax.legend(loc="best", title="layer")
	fig.savefig(f'{out_dir}/plot_s-values_PR.png', bbox_inches="tight")
	plt.close(fig)

	# ALL SINGULAR VALUES
	cols=n_layers
	fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
	if cols == 1:
	    axs = [axs]
	fig.suptitle(title)
	for l, S in enumerate(Ss):
	    ax = axs[l]
	    ax.set_title(rf"$W_{l+1}$")
	    ax.set_xlabel('epoch')
	    ax.set_ylabel('singular value')
	    for s in S.T:
	        ax.plot(epochs, s)
	fig.savefig(f'{out_dir}/plot_s-values.png', bbox_inches="tight")
	plt.close(fig)

	# SINGULAR VALUES DISTRIBUTION
	cols=n_layers-1
	fig, axs = plt.subplots(1, cols, figsize=(cols*4, 4))
	if cols == 1:
	    axs = [axs]
	fig.suptitle(title)
	for l, S in enumerate(Ss[:-1]):
	    ax = axs[l]
	    ax.set_title(rf"$W_{l+1}$")
	    ax.set_xlabel('singular value')
	    ax.set_ylabel('density')
	    ax.hist(S[0], density=True, bins=30, label="initial", alpha=0.4)
	    ax.hist(S[-1], density=True, bins=30, label="trained", alpha=0.4)
	    ax.legend(loc="best")
	fig.savefig(f'{out_dir}/plot_eval_distr.png', bbox_inches="tight")
	plt.close(fig)


def plot_loss_accuracy (train_loss, test_loss, train_acc=None, test_acc=None, epochs=None, out_dir='.', title=''):

	n_snapshots = len(train_loss)
	if epochs is None:
		epochs = np.arange(n_snapshots)

	# TRAIN AND TEST LOSS
	fig, ax = plt.subplots(figsize=(6, 4))
	# ax.scatter(np.arange(len(test_loss)), test_loss, label="test", s=2, c="C1")
	# ax.plot(train_loss, label="train", c="C0")
	ax.scatter(epochs, test_loss, label="test", s=2, c="C1")
	ax.plot(epochs, train_loss, label="train", c="C0")
	ax.set_title(title)
	ax.grid()
	# ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_ylabel('Train and test loss')
	ax.set_xlabel('epoch')
	ax.legend(loc="best")
	fig.savefig(f'{out_dir}/plot_loss.png', bbox_inches="tight")
	plt.close(fig)

	if (train_acc is not None) and (test_acc is not None):
		# TRAIN AND TEST ACCURACY
		fig, ax = plt.subplots(figsize=(6, 4))
		# ax.scatter(np.arange(len(test_acc)), test_acc, label="test", s=2, c="C1")
		# ax.plot(train_acc, label="train", c="C0")
		ax.scatter(epochs, test_acc, label="test", s=2, c="C1")
		ax.plot(epochs, train_acc, label="train", c="C0")
		ax.set_title(title)
		ax.grid()
		# ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_ylabel('Train and test accuracy')
		ax.set_xlabel('epoch')
		ax.legend(loc="best")
		fig.savefig(f'{out_dir}/plot_accuracy.png', bbox_inches="tight")
		plt.close(fig)


def plot_weights (model_weights, weights_norm, epochs=None, out_dir='.', title=''):

	n_layers = len(model_weights)
	n_snapshots = len(model_weights[0])
	if epochs is None:
		epochs = np.arange(n_snapshots)

	# NORM OF THE WEIGHTS
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(title)
	ax.set_ylabel('L2 weight norm')
	ax.grid()
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('epoch')
	# ax.set_ylim([0,1])
	colors = ['C0', 'C1', 'C2', 'C3']
	for i, (norm, c) in enumerate(zip(weights_norm, colors)):
	    ax.plot(epochs, norm/norm[0], c=c, label=f'{i+1}: {norm[0]:.2f}')
	ax.legend(loc='best', title="layer: init value")
	fig.savefig(f'{out_dir}/plot_weights_norm.png', bbox_inches="tight")
	plt.close(fig)

	# HISTOGRAM OF THE WEIGHTS
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(title)
	ax.set_xlabel('L2 weight norm (trained)')
	ax.set_ylabel('density')
	N = np.max([m.shape[1] for m in model_weights])
	ax.set_xlim([-3/np.sqrt(N),3/np.sqrt(N)])
	for l, W in enumerate(model_weights):
	    ax.hist(W[-1].ravel(), density=True, bins=100, label=f"W_{l+1}", alpha=0.3)
	ax.legend(loc="best")
	fig.savefig(f'{out_dir}/plot_weights_histogram.png', bbox_inches="tight")
	plt.close(fig)


def plot_hidden_units (hidden, epochs=None, out_dir='.', title=''):

	n_layers = len(hidden) + 1
	n_snapshots = len(hidden[0])
	if epochs is None:
		epochs = np.arange(n_snapshots)

	# VARIANCE OF THE HIDDEN LAYER
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(title)
	ax.set_ylabel('Hidden layer norm')
	ax.set_xlabel('epoch')
	ax.grid()
	for l, h in enumerate(hidden):
	    ax.plot(epochs, np.linalg.norm(h, axis=1), label=f"X_{l+1}")
	ax.legend(loc="best", title="hidden layer")
	fig.savefig(f'{out_dir}/plot_hidden_layer_norm.png', bbox_inches="tight")
	plt.close(fig)

	# HISTOGRAM OF THE HIDDEN LAYER(S)
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_title(title)
	ax.set_xlabel('Hidden layer activity')
	ax.set_ylabel('density')
	for i, h in enumerate(hidden):
	    ax.hist(h[-1], density=True, bins="sqrt", label=f"X_{l+1}", alpha=0.3)
	ax.legend(loc="best", title="hidden layer")
	fig.savefig(f'{out_dir}/plot_hidden_layer_histogram.png', bbox_inches="tight")
	plt.close(fig)
