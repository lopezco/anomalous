import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as scale_data


def anomaly(x, n=10, method="hdr", robust=False, plot=True, labels=True, col=None):
    """
    :param x: a pd.DataFrame returned by `ts_measures` function 
    :param n: 
    :param method: 
    :param robust: 
    :param plot: 
    :param labels: 
    :param col: 
    :return: 
    """
    nc = len(x)
    if nc < n:
        raise ValueError("Your n is too large.")

    X = x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all').dropna(axis=0, how='any')

    # robust PCA space (scaling version)
    X = scale_data(X, with_mean=True, with_std=True)

    if robust:
        raise NotImplemented('Robust PCA has not been implemented yet')
    else:
        pca = PCA(n_components=2)
        pca.fit(X)
        scores = pca.transform(X)

    idx = np.zeros(n)

    if method == "hdr":
        # TODO: implement Bivariate Highest Density Regions computation
        # hdrinfo = hdrcde::hdr.2d(x=scores[:, 0], y=scores[:, 1], kde.package="ks")
        # idx = sorted(hdrinfo$fxy)[1:n]
        idx = range(idx)
        main = "Lowest densities on anomalies"
    else:
        raise NotImplemented('Alpha hull using binary split has not been implemented yet')

    if plot:
        if col is None:
            col = ("#000000", "darkred")
        else:
            col = [col] if not isinstance(col, (list, tuple)) else col
            col = np.unique(col)

            if len(col) == 1:
                col = np.repeat(col, 2)
            else:
                col = np.unique(col)[0:2]

        is_outlier = np.zeros(len(X)).astype(bool)
        is_outlier[idx] = True

        fig, ax = plt.subplots()
        ax.scatter(x=scores[~is_outlier, 0], y=scores[~is_outlier, 1], c=col[0])
        ax.scatter(x=scores[is_outlier, 0], y=scores[is_outlier, 1], c=col[1], marker='x')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(main)

        if labels:
            for txt, i_data in enumerate(idx):
                ax.annotate(s=txt, xy=(scores[i_data, 0], scores[i_data, 1]))
        plt.show()

    return dict(index=idx, scores=scores)
