# -*- coding: utf-8 -*-

# Import(s)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def distribution (y, var, X=None, xlabel='', ylabel='', legend=True, bins=np.linspace(0,1,100,endpoint=True), ax=None):
    """
    ...
    """
    plt.rcParams.update({'font.size': 20})
    # Check(s)
    if isinstance(var, int):
        assert X is not None, "Requested plot of integer variable with no feature array."
        var = X[:,var]
        pass

    # Define variable(s)
    sig = (y == 1)
    common = dict(bins=bins)

    # Ensure axes exist
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass

    # Get weights
    nsig = np.sum( sig)
    nbkg = np.sum(~sig)
    wsig = np.ones((nsig,)) / float(nsig)
    wbkg = np.ones((nbkg,)) / float(nbkg)

    # Plot distributions
    ax.hist(var[ sig], weights=wsig, color='r', histtype='step', linewidth = 2, label='Signal',     **common)
    ax.hist(var[~sig], weights=wbkg, color='b', histtype='step', linewidth = 2, label='Background', **common)

    # Decorations
    ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    ax.set_xlim((0,1))
    ax.set_ylabel(ylabel or 'Fraction of Events')
    if legend:
        ax.legend()
        pass

    return ax


def profile (m, ys, labels=None, bins=np.linspace(100,150,50,endpoint=True), ax=None):
    """
    ...
    """
    plt.rcParams.update({'font.size': 20})
    # Check(s)
    if isinstance(bins, int):
        bins = np.linspace(m.min(), m.max(), bins + 1, endpoint=True)
        pass

    if not isinstance(ys, list):
        ys = [ys]
        pass

    N = len(ys)
    centres = bins[:-1] + 0.5 * np.diff(bins)

    if labels is None:
        labels = [None for _ in range(N)]
    elif isinstance(labels, str):
        labels = [labels]
        pass

    assert len(labels) == N, "[profile] Number of observables ({}) and associated labels ({}) do not match.".format(N, len(labels))

    # Local background efficiency
    profiles = {ix: list() for ix in range(N)}
    means_NN  = list()
    means_ANN = list()
    for down, up in zip(bins[:-1], bins[1:]):
        msk = (m >= down) & (m < up)
        for ix, y in enumerate(ys):
            profiles[ix].append(y[msk].mean())
            pass
        pass

    # Ensure axes exist
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass

    # Plot profile(s)
    for ix in range(N):
        ax.plot(centres, profiles[ix], '.-', label=labels[ix])
        pass

    # Decorations
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('Average Value')
    ax.set_ylim((0,1))
    ax.set_xlim(bins[0], bins[-1])
    ax.legend()

    return ax


def sculpting (m, y, preds, labels=None, effsig = 0.5, bins=40, ax=None):
    """
    ...
    """
    plt.rcParams.update({'font.size': 20})
    # Check(s)
    if isinstance(bins, int):
        bins = np.linspace(100, 150, 50, endpoint=True)
        pass

    if not isinstance(preds, list):
        preds = [preds]
        pass

    N = len(preds)

    if labels is None:
        labels = [None for _ in range(N)]
    elif isinstance(labels, str):
        labels = [labels]
        pass

    assert len(labels) == N, "[sculpting] Number of observables ({}) and associated labels ({}) do not match.".format(N, len(labels))

    # ... labels...

    # Ensure axes exist
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass

    # Common definitions
    sig = (y == 1).ravel()
    common = dict(bins=bins, alpha=0.5)

    # Get weights
    nsig = np.sum( sig)
    nbkg = np.sum(~sig)
    wsig = np.ones((nsig,)) / float(nsig)
    wbkg = np.ones((nbkg,)) / float(nbkg)

    # Draw original mass spectrum, legend header
    ax.hist(m[ sig], color='red',  bins=bins, weights=wsig, label='Signal', histtype='step', lw=2)
    ax.hist(m[~sig], color='blue', bins=bins, weights=wbkg, label='Background', histtype='step', lw=2)
    # ax.hist([0], weights=[0], color='black', label='Bkgds., $\\varepsilon_{{sig}} = {:.0f}\%$ cut'.format(effsig * 100.), **common)

    # Draw post-cut mass spectra
    for pred, label in zip(preds, labels):

        # -- Get cut
        cut = np.percentile(pred[sig], effsig * 100.)
        msk = (~sig) & (pred > cut).ravel()  # Assuming signal towards larger values

        # -- Get weights
        nmsk = np.sum( msk)
        wmsk = np.ones((nmsk,)) / float(nmsk)

        # -- Plot
        ax.hist(m[msk],  weights=wmsk, label="   " + label, **common)
        pass

    ax.set_ylabel('Fraction of Events /GeV')
    ax.set_xlabel('Mass [GeV]')
    ax.set_xlim((100, 150))
    ax.legend()

    return ax


def posterior (adv, m, z, nb_bins=np.linspace(90, 180, 50, endpoint=True), title=''):
    """
    ...
    Arguments:
        M: Jet masses
        Z: Classifier outputs
        nb_bins: ...
        title: ...
    Returns:
        fig, ax: ...
    """
    plt.rcParams.update({'font.size': 20})
    # Definitions
    scale = m.max()
    colours = ['r', 'g', 'b']
    zs = [0.2, 0.4, 0.8]
    tol = 0.05


    # Binning, scaled and not
    mt_pdf = np.linspace(0, 1., 1000 + 1, endpoint=True)
    bins = mt_pdf * scale

    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    # -- Left pane
    # Draw inclusive background jet distribution
    ax[0].hist(m, nb_bins, density=1., alpha=0.3, color='black', label='Background')
    for col, z_ in zip(colours, zs):
        # Draw adversary posterior p.d.f. for classifier output `z`
        posterior = adv.predict([z_*np.ones_like(mt_pdf), mt_pdf])
        ax[0].plot(mt_pdf * scale, posterior / scale, color=col, label='z = {:.1f}'.format(z_))
        pass

    # -- Right pane
    ax[1].hist([0], nb_bins, alpha=0.3, weights=[0], color='black', label='Background, $z_{NN}$-bin.')
    for col, z_ in zip(colours, zs):
        # Draw background jet distribution for classifier output `z`
        msk = np.abs(z - z_).ravel() < tol

        ax[1].hist(m[msk], nb_bins, color=col, density=1., alpha=0.3, label='  {:.2f} < $z_{{NN}}$ < {:.2f}'.format(z_ - tol, z_ + tol))

        # Draw adversary posterior p.d.f. for classifier output `z`
        posterior = adv.predict([z_*np.ones_like(mt_pdf), mt_pdf])
        ax[1].plot(mt_pdf * scale, posterior / scale, color=col, label='z = {:.1f}'.format(z_))
        pass

    # -- Decorations
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('Probability density')
    ax[0].set_xlabel('Mass [GeV]')
    ax[1].set_xlabel('Mass [GeV]')
    ax[0].set_xlim((90,180))
    ax[1].set_xlim((90,180))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return ax