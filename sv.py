import argparse
import pickle
import numpy
import theano
from theano import tensor
from pylearn2.autoencoder import ContractiveAutoencoder
from numpy.linalg import svd
from video import VideoSink


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.

    http://www.scipy.org/Cookbook/RankNullspace
    """

    A = numpy.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def norm(x):

    x -= x.min()
    max_val = x.max()
    if max_val == 0:
        max_val = 0.000001
    x /= max_val

    return x

def hessian(model, h_indx, data):
    """
    Return the hessian values

    Parameters
    ----------
    model : model object
    h_indx : Index of the unit
    data : data array
    """

    x = tensor.vector()
    h = theano.gradient.hessian(model.encode(x)[h_indx], x)

    fn = theano.function(inputs = [x], outputs = h)

    return fn(data)



def grad_activation(model, h_indx, data):

    x = tensor.vector()
    j = theano.gradient.Rop(model.encode(x)[h_indx], x, x)

    fn = theano.function(inputs = [x], outputs = j)

    return fn(data)

def get_optimal_stimuli(model, hid_indx, img_shape, learning_rate, n_epochs):
    """
    Find the stimlui that maximazie each hidden unit activation, use sgd
    to find the optimal value starting from unit sphere

    Parameters
    ----------

    model : model object
    hid_indx : unit index
    img_shaoe : Image shape
    learning_rate : SGD learning rate
    n_epochs : SGD number of epochs
    """

    inputs = theano.shared(numpy.ones(numpy.prod(img_shape), dtype = theano.config.floatX))

    x = tensor.vector()
    hid_val = model.encode(x)[hid_indx]
    grad = tensor.grad(-hid_val, x)
    updates = {inputs : inputs - grad * learning_rate}

    sgd_f = theano.function(inputs = [], outputs = hid_val, updates = updates, givens = {x : inputs})
    for epoch in xrange(n_epochs):
        res = sgd_f()
        print "\tepoch %d, activation: %f " %(epoch, res)

    x_optimal = inputs.get_value()
    x_optimal = norm(x_optimal)

    return x_optimal


def visualize(model,
                nhid,
                image_shape,
                learning_rate,
                n_epochs,
                num_dir = 3,
                cutoff_percent = 0.7,
                save_path = '.',
                f_name = 'network'):



    for i in xrange(nhid):
        print "Hidden unit: %d" %(i)

        opt_x = get_optimal_stimuli(model, i, image_shape, learning_rate, n_epochs)
        tangent_basis = nullspace(opt_x.T)
        h = hessian(model, i, opt_x)
        h = numpy.dot(tangent_basis.T, numpy.dot(h, tangent_basis))
        d, v = numpy.linalg.eig(h)
        test = numpy.real(d)
        order = numpy.argsort(-test)
        order = order[numpy.hstack([numpy.arange(num_dir), numpy.arange(len(order) - num_dir, len(order))])]
        v = numpy.real(v[:,order])
        opt_act = grad_activation(model, i, opt_x)
        cutoff = cutoff_percent * opt_act

        for j in xrange(len(order)):
            movie = []
            start = 0.
            stop = 0.
            while start >= -numpy.pi / 2.:
                curr_x = opt_x * numpy.cos(start) + numpy.sin(start) * numpy.dot(tangent_basis, v[:,j])
                curr_act = grad_activation(model, i, curr_x)
                if curr_act < cutoff:
                    start += numpy.pi / 18.
                    break
                start -= numpy.pi / 18.
            while stop <= numpy.pi / 2.:
                curr_x = opt_x * numpy.cos(stop) + numpy.sin(stop) * numpy.dot(tangent_basis, v[:,j])
                curr_act = grad_activation(model, i, curr_x)
                if curr_act < cutoff:
                    stop -= numpy.pi / 18.
                    break
                stop += numpy.pi / 18.

            for t in numpy.arange(start, stop, numpy.pi / 36):
                curr_x = opt_x * numpy.cos(t) + numpy.sin(t) * numpy.dot(tangent_basis, v[:, j])
                movie.append(curr_x.reshape(image_shape))

            if len(movie) == 0:
                break

            movie = numpy.asarray(movie)
            movie = norm(movie)
            movie = numpy.int8(255 * movie)

            if j <= num_dir:
                fname = "%s/%s_unit_%d_eig_high_%d" %(save_path, f_name, i, j)
            else:
                fname = "%s/%s_unit_%d_eig_high_%d" %(save_path, f_name, i, j - num_dir)
            save_movie(movie, image_shape, fname)


def save_movie(movie, image_shape, name):
    """ Save numpy array as avi movie """
    rate = 5
    video = VideoSink(size = image_shape, filename = name, rate = rate,  byteorder="y8")
    for frame in movie:
        video.run(frame)
    video.close()

def load_model(data_path):
    """
    loads model and return it with number if hidden units
    """

    data = pickle.load(open(data_path, 'r'))

    return data, data.nhid

def dummy_test():

    image_shape = (28, 28)
    cae = ContractiveAutoencoder(nvis = 28*28, nhid = 10, act_enc = "sigmoid", act_dec = "sigmoid")
    visualize(cae, nhid = 10, image_shape = image_shape, learning_rate = 0.1, n_epochs = 30, save_path = 'videos')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Visualize stimuli's")
    parser.add_argument('-m', '--model', help = 'path to model file',
            required = True)
    parser.add_argument('-o', '--output', help = 'path to save videos',
            default = 'videos/')
    parser.add_argument('-n', '--name', default = 'test', help = 'experiment name')
    parser.add_argument('-W', '--width', help = 'image width',
            required = True, type = int)
    parser.add_argument('-H', '--height', help = 'image height',
            required = True, type = int)
    parser.add_argument('-l', '--learning_rate', help = "learning rate",
            default = 0.1, type = float)
    parser.add_argument('-e', '--epochs', help = "number of epochs",
            default = 30, type = int)
    parser.add_argument('-d', '--directions', help = "number of directions",
            default = 3, type = int)
    parser.add_argument('-c', '--cutoff', help = "cutoff percent",
            default = 0.7, type = float)
    args = parser.parse_args()

    model, nhid = load_model(args.model)

    visualize(model, nhid = nhid,
            image_shape = (args.width, args.height),
            learning_rate = args.learning_rate,
            n_epochs = args.epochs,
            save_path = args.output,
            f_name = args.name)


    #dummy_test()
