import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os.path import join
plt.rcParams['font.size'] = 10

def plot_alignment_layers (projs, d_output=10, epochs=None, out_dir='.', title=''):

    n_layers = len(projs) + 1
    n_snapshots = len(projs[0])
    if epochs is None:
        epochs = np.arange(n_snapshots)
    n_skip = epochs[1]-epochs[0]

    # ALIGNMENT
    kwargs=dict(vmin=0, vmax=1, aspect='equal') # cmap="bwr", vmin=-1, 
    cols=len(projs)
    fig, axs = plt.subplots(1, cols, figsize=(cols*3, 2))
    if cols == 1:
        axs = [axs]
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.5)
    def plot_frame (frame):
        plt.cla()
        # fig.suptitle(title+f" -- epoch {frame*n_skip}")
        # plot alignment of intermediate layers
        for l, proj in enumerate(projs):
            ax = axs[l]
            ax.set_xlabel(r"$m$")
            ax.set_ylabel(r"$n$")
            ax.set_title(rf"$|V^n_{l+2}\cdot U^m_{l+1}$|")
            im = ax.imshow(np.abs(proj[frame, :d_output+4, :d_output+4]), **kwargs)#; plt.colorbar(im, ax=ax)

    plot_frame(len(projs[0])-1)
    fig.savefig(join(out_dir, 'plot_alignment_layers_final.png'), bbox_inches="tight")

    duration = 10 # in s
    n_frames = 50 # total number of frames to plot
    dt = duration / n_frames * 1000. # in ms
    frames = np.linspace(0,len(projs[0])-1,n_frames).astype(int)
    ani = FuncAnimation(fig, plot_frame,
                        interval=dt,
                        frames=frames,
                        blit=False)
    ani.save(join(out_dir, 'plot_alignment_layers.gif'))

    cols=n_layers-1
    fig, axs = plt.subplots(1, cols, figsize=(cols*3, 2))
    if cols == 1:
        axs = [axs]
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.5)
    # fig.suptitle(title)
    for l, proj in enumerate(projs):
        ax = axs[l]
        ax.set_ylim([0,1.1])
        ax.set_xlabel("epoch")
        ax.set_ylabel(rf"$|V^n_{l+2}\cdot U^m_{l+1}|$")
        _,_n,_m = proj.shape
        n = min(_n,d_output+2)
        m = min(_m,d_output+2)
        for i in range(n):
            for j in range(m):
                c = "C0" if i == j else "C1"
                ax.plot(epochs, np.abs(proj[:, i, j]), c=c)
    fig.savefig(join(out_dir, 'plot_alignment_layers_epochs.png'), bbox_inches="tight")


def plot_alignment_wstar (model_weights, w_star, Us,Vs, epochs=None, out_dir='.', title=''):
    
    n_layers = len(model_weights)
    n_snapshots = len(model_weights[0])
    d_output = model_weights[-1].shape[1]
    if epochs is None:
        epochs = np.arange(n_snapshots)

    Uw, Sw, Vhw = np.linalg.svd(np.atleast_2d(w_star))
    overlaps = [
        np.dot(Vs[0], Vhw.T), # overlap btw right modes
        np.dot(Us[-1], Uw.T), # overlap btw left modes
    ]

    # BIMODALITY
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
    ax.set_xlabel(r'$W^{(L)}_j$')
    ax.set_ylabel(r'$(W^{(1)}\cdot w^*)_j$')
    W1_dot_wstar = np.dot(np.atleast_2d(w_star), model_weights[0][-1].T)
    for i in range(d_output):
        ax.scatter(model_weights[-1][-1,i], W1_dot_wstar[i], alpha=0.5, s=.1)
    fig.savefig(join(out_dir, 'plot_scatter_W.png'), bbox_inches="tight")
    plt.close(fig)

    # COS OF ANGLE BETWEEN PRINCIPAL COMPOMENTS AND WEIGHTS
    # (check low rank of W)
    fig, axs = plt.subplots(1,2,figsize=(6, 2))
    plt.subplots_adjust(wspace=0.4)
    # fig.suptitle(title)
    for ax in axs.ravel():
        ax.set_xlabel('epoch')
        ax.set_ylim([0,1.1])
        ax.grid()
    axs[0].set_ylabel(r'$|\cos\theta(\tilde{V}, V_1)|$')
    axs[1].set_ylabel(r'$|\cos\theta(\tilde{U}, U_L)|$')
    for i in range(2):
        _,n,m = overlaps[i].shape
        n = min(n, d_output+2)
        m = min(m, d_output+2)
        for j in range(n):
            axs[i].plot(epochs, np.abs(overlaps[i])[:,j,j], c=f'C{j}')
            for k in range(j+1,m):
                axs[i].plot(epochs, np.abs(overlaps[i])[:,j,k], c=f'C{j}', ls='--')
    fig.savefig(join(out_dir, 'plot_alignment_wstar.png'), bbox_inches="tight")
    plt.close(fig)


def plot_singular_values (Ss, epochs=None, out_dir='.', title='',
    xlim=None):

    n_layers = len(Ss)
    n_snapshots = len(Ss[0])
    if epochs is None:
        epochs = np.arange(n_snapshots)

    # PARTICIPATION RATIO AND LARGEST SINGULAR VALUE
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('participation ratio / rank')
    ax.grid()
    PR = lambda S: np.array([np.sum(s)**2/np.sum(s**2)/len(s) for s in S])
    for l, S in enumerate(Ss):
        ax.plot(epochs, PR(S), label=f"{l+1}")
    ax.legend(loc="best", title="layer")
    fig.savefig(join(out_dir, 'plot_s-values_PR.png'), bbox_inches="tight")
    plt.close(fig)

    # ALL SINGULAR VALUES
    cols=n_layers
    fig, axs = plt.subplots(1, cols, figsize=(cols*3, 2))
    plt.subplots_adjust(wspace=0.4)
    # fig.suptitle(title)
    for l, S in enumerate(Ss):
        ax = axs[l]
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(rf"$W_{l+1}$")
        ax.set_xlabel('epoch')
        ax.set_ylabel('singular value')
        for s in S.T:
            ax.plot(epochs, s)
    fig.savefig(join(out_dir, 'plot_s-values.png'), bbox_inches="tight")
    plt.close(fig)

    # SINGULAR VALUES DISTRIBUTION
    cols=n_layers-1
    fig, axs = plt.subplots(1, cols, figsize=(cols*3, 2))
    plt.subplots_adjust(wspace=0.4)
    if cols == 1:
        axs = [axs]
    # fig.suptitle(title)
    for l, S in enumerate(Ss[:-1]):
        ax = axs[l]
        ax.set_title(rf"$W_{l+1}$")
        ax.set_xlabel('singular value')
        ax.set_ylabel('density')
        ax.hist(S[0], density=True, bins=30, label="initial", alpha=0.4)
        ax.hist(S[-1], density=True, bins=30, label="trained", alpha=0.4)
        ax.legend(loc="best")
    fig.savefig(join(out_dir, 'plot_eval_distr.png'), bbox_inches="tight")
    plt.close(fig)


def plot_loss_accuracy (train_loss, test_loss, train_acc=None, test_acc=None,
        train_epochs=None, test_epochs=None, out_dir='.', title='',
        yscale=None, xscale=None, xlim=None):

    train_snapshots = len(train_loss)
    if train_epochs is None:
        train_epochs = np.arange(train_snapshots)

    test_snapshots = len(test_loss)
    if test_epochs is None:
        test_epochs = np.arange(test_snapshots)

    # TRAIN AND TEST LOSS
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.scatter(np.arange(len(test_loss)), test_loss, label="test", s=2, c="C1")
    # ax.plot(train_loss, label="train", c="C0")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.scatter(test_epochs, test_loss, label="test", s=2, c="C1")
    ax.plot(train_epochs, train_loss, label="train", c="C0")
    # ax.set_title(title)
    ax.grid()
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_ylabel('Train and test loss')
    ax.set_xlabel('epoch')
    ax.legend(loc="best")
    fig.savefig(join(out_dir, 'plot_loss.png'), bbox_inches="tight")
    plt.close(fig)

    if (train_acc is not None) and (test_acc is not None):
        # TRAIN AND TEST ACCURACY
        fig, ax = plt.subplots(figsize=(3, 2))
        # ax.scatter(np.arange(len(test_acc)), test_acc, label="test", s=2, c="C1")
        # ax.plot(train_acc, label="train", c="C0")
        ax.scatter(test_epochs, test_acc, label="test", s=2, c="C1")
        ax.plot(train_epochs, train_acc, label="train", c="C0")
        # ax.set_title(title)
        ax.grid()
        ax.set_ylabel('Train and test accuracy')
        ax.set_xlabel('epoch')
        ax.legend(loc="best")
        fig.savefig(join(out_dir, 'plot_accuracy.png'), bbox_inches="tight")
        plt.close(fig)


def plot_weights (model_weights, weights_norm, epochs=None, out_dir='.', title=''):

    n_layers = len(model_weights)
    n_snapshots = len(model_weights[0])
    if epochs is None:
        epochs = np.arange(n_snapshots)

    # NORM OF THE WEIGHTS
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
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
    fig.savefig(join(out_dir, 'plot_weights_norm.png'), bbox_inches="tight")
    plt.close(fig)

    # HISTOGRAM OF THE WEIGHTS
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
    ax.set_xlabel('L2 weight norm (trained)')
    ax.set_ylabel('density')
    N = np.max([m.shape[1] for m in model_weights])
    ax.set_xlim([-3/np.sqrt(N),3/np.sqrt(N)])
    for l, W in enumerate(model_weights):
        ax.hist(W[-1].ravel(), density=True, bins=100, label=f"W_{l+1}", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(join(out_dir, 'plot_weights_histogram.png'), bbox_inches="tight")
    plt.close(fig)


def plot_hidden_units (hidden, epochs=None, out_dir='.', title=''):

    n_layers = len(hidden) + 1
    n_snapshots = len(hidden[0])
    if epochs is None:
        epochs = np.arange(n_snapshots)

    # VARIANCE OF THE HIDDEN LAYER
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
    ax.set_ylabel('Hidden layer norm')
    ax.set_xlabel('epoch')
    ax.grid()
    for l, h in enumerate(hidden):
        ax.plot(epochs, np.linalg.norm(h, axis=1), label=f"X_{l+1}")
    ax.legend(loc="best", title="hidden layer")
    fig.savefig(join(out_dir, 'plot_hidden_layer_norm.png'), bbox_inches="tight")
    plt.close(fig)

    # HISTOGRAM OF THE HIDDEN LAYER(S)
    fig, ax = plt.subplots(figsize=(3, 2))
    # ax.set_title(title)
    ax.set_xlabel('Hidden layer activity')
    ax.set_ylabel('density')
    for i, h in enumerate(hidden):
        ax.hist(h[-1], density=True, bins="sqrt", label=f"X_{l+1}", alpha=0.3)
    ax.legend(loc="best", title="hidden layer")
    fig.savefig(join(out_dir, 'plot_hidden_layer_histogram.png'), bbox_inches="tight")
    plt.close(fig)


def plot_covariance (cov, d_output=1, IO=False, W_product=None, out_dir='.', title=''):

    cov_XX = cov
    if IO:
        d_input = len(cov) - d_output
        cov_XX = cov[:d_input,:d_input]
        cov_Xy = cov[:d_input,-d_output:]
        cov_yy = cov[-d_output:,-d_output:]

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    plt.subplots_adjust(wspace=0.4)
    ax = axs[0]
    ax.set_title("Covariance matrix")
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    im = ax.imshow(cov_XX)
    fig.colorbar(im, ax=ax)
    
    ax = axs[1]
    ax.set_title("Covariance spectrum")
    ax.set_yscale("log")
    ax.set_ylim([1e-5,1])
    ax.set_xlabel(r'mode, $n$')
    ax.set_ylabel(r'$\sqrt{\lambda_n}\,/\,N$')
    S, _ = np.linalg.eig(cov_XX)
    idx_sorted = np.argsort(S)[::-1]
    S_sorted = np.abs(S[idx_sorted]) # to remove small imaginary parts
    ax.plot(np.sqrt(S_sorted)/len(cov_XX))
    
    axins = ax.inset_axes([0.15, 0.15, 0.4, 0.4])
    axins.set_ylim([0,1])
    _n = 20 # min(d_output + 2, len(cov_XX))
    axins.plot( np.sqrt(S_sorted[:_n]) / len(cov_XX) )
    fig.savefig(join(out_dir, 'plot_input_covariance.png'), bbox_inches="tight")
    plt.close(fig)


    if IO:

        U, S, Vh = np.linalg.svd(cov_Xy)
        if W_product is not None:
            # calculate the input-output covariance matrix given the product of the weights
            # cov_Xy_l = np.einsum('...ij,jk->...ik', W_product, cov_XX).transpose((0,2,1))
            cov_Xy_l = W_product
            U_l, S_l, Vh_l = np.linalg.svd(cov_Xy_l)
            fig, axs = plt.subplots(2,4,figsize=(10,7))
        else:
            fig, axs = plt.subplots(1,4,figsize=(10,3))
        plt.tight_layout()
        ax = axs.ravel()

        kwargs = dict(vmin=-1, vmax=1, cmap="bwr", aspect="equal")

        def plot_frame(frame):
            for a in ax[:4]:
                a.clear()
            # plot the input-output covariance matrix of the (training) data
            ax[0].set_ylabel("Training data")
            im = ax[0].imshow(cov_Xy, **kwargs); ax[0].set_title(r"$\Sigma^{xy}$")
            # plot the left singular vectors
            ax[1].imshow(-U[:,:d_output], **kwargs); ax[1].set_title(r"$U$")
            # plot the diagonal singular-value matrix
            ax[2].imshow(np.diag(S), aspect="equal"); ax[2].set_title(r"$S$")
            for i,s in enumerate(S):
                ax[2].text(i,i,f"{s:.2f}", c='r',verticalalignment="center",horizontalalignment="center")
            # plot the right signular vectors
            ax[3].imshow(-Vh, **kwargs); ax[3].set_title(r"$V^T$")

            if W_product is not None:
                for a in ax[4:]:
                    a.clear()
                # plot the input-output covariance matrix from the model
                ax[4].set_ylabel("Model")
                im = ax[4].imshow(cov_Xy_l[frame], **kwargs); ax[4].set_title(r"$\Sigma^{xy}$")
                # plot the left singular vectors
                ax[5].imshow(-U_l[frame,:,:d_output], **kwargs); ax[5].set_title(r"$U$")
                # plot the diagonal singular-value matrix
                ax[6].imshow(np.diag(S_l[frame]), aspect="equal"); ax[6].set_title(r"$S$")
                for i,s in enumerate(S_l[frame]):
                    ax[6].text(i,i,f"{s:.2f}", c='r',verticalalignment="center",horizontalalignment="center")
                # plot the right signular vectors
                ax[7].imshow(-Vh_l[frame], **kwargs); ax[7].set_title(r"$V^T$")

        im = plot_frame(-1)
        # fig.colorbar(im, ax=ax[4:].ravel().tolist())
        # fig.colorbar(im, ax=ax[:4].ravel().tolist())
        fig.savefig(join(out_dir, 'plot_input-output_covariance.png'), bbox_inches="tight")

        duration = 10 # in s
        n_frames = 100 # total number of frames to plot
        dt = duration / n_frames * 1000. # in ms
        frames = np.linspace(0,len(cov_Xy_l)-1,n_frames).astype(int) # indices of frames to plot
        ani = FuncAnimation(fig, plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
    ani.save(join(out_dir, 'plot_input-output_covariance.gif'))
