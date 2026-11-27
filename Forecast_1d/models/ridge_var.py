"""Direct multi-output ridge VAR: both fluxes, lagged history, all leads."""
import numpy as np


def windows(values, origins, in_len):
    """Construct history and target windows for the supplied forecast origins."""
    return np.stack([values[t-in_len+1:t+1] for t in origins]).astype(np.float64)


def fit_ridge(values, train_end, in_len, horizon, alpha=0.01):
    """Minimize mean squared normalized residuals + alpha * squared weights."""
    if alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    values = np.asarray(values,dtype=np.float64)
    train = values[:train_end]
    origins = np.arange(in_len-1,train_end-horizon)
    if not len(origins):
        raise ValueError('No complete ridge training windows')
    result = []
    for p in range(values.shape[1]):
        raw = train[:,p,:]
        finite = np.where(np.isfinite(raw),raw,np.nan)
        mean,scale = np.nanmean(finite,axis=0),np.nanstd(finite,axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError('No finite training data')
        scale = np.where(scale<1e-8,1,scale)
        x = windows(raw,origins,in_len)
        y = np.stack([raw[t+1:t+horizon+1] for t in origins])
        keep = np.isfinite(y).all(axis=(1,2))
        if not keep.any():
            raise ValueError('No finite ridge training targets')
        x = ((np.where(np.isfinite(x),x,mean)-mean)/scale)[keep].reshape(keep.sum(),-1)
        y = ((y[keep]-mean)/scale).reshape(keep.sum(),-1)
        xm,ym = x.mean(axis=0),y.mean(axis=0)
        xc,yc = x-xm,y-ym
        weights = np.linalg.solve(xc.T@xc + alpha*len(x)*np.eye(x.shape[1]),xc.T@yc)
        result.append((mean,scale,weights,ym-xm@weights))
    return result


def forecast_ridge(values, origins, in_len, horizon, fitted):
    """Forecast ridge for the ridge var workflow."""
    output=[]
    for p,(mean,scale,weights,bias) in enumerate(fitted):
        x=windows(values[:,p,:],origins,in_len)
        x=((np.where(np.isfinite(x),x,mean)-mean)/scale).reshape(len(origins),-1)
        y=(x@weights+bias).reshape(len(origins),horizon,2)*scale+mean
        output.append(y.transpose(0,2,1))
    return np.stack(output,axis=1)
