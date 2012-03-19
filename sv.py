import subprocess
import numpy
import theano
from theano import tensor
from pylearn2.autoencoder import ContractiveAutoencoder
from numpy.linalg import svd

class VideoSink(object) :
    """
    credit: https://github.com/vokimon/freenect_python_processing/blob/master/src/videosink.py
    """
    def __init__( self, size, filename="output", rate=1, byteorder="bgra" ) :
        self.size = size
        cmdstring  = ('mencoder',
            '/dev/stdin',
            '-demuxer', 'rawvideo',
            '-rawvideo', 'w=%i:h=%i'%size[::-1]+":fps=%i:format=%s"%(rate,byteorder),
            '-o', filename+'.avi',
            '-ovc', 'lavc',
            )
        self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def run(self, image) :
        assert image.shape == self.size
        #image.swapaxes(0,1).tofile(self.p.stdin) # should be faster but it is indeed slower
        self.p.stdin.write(image.tostring())
    def close(self) :
        self.p.stdin.close()



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


def hessian(model, h_indx, data):

    x = tensor.vector()
    h = theano.gradient.hessian(model.encode(x)[h_indx], x)

    fn = theano.function(inputs = [x], outputs = h)

    return fn(data)


def activation(model, h_indx, data):

    x = tensor.vector()

    fn = theano.function(inputs = [x], outputs = model.encode(x)[h_indx])

    return fn(data)

def get_optimal_stimuli(model, hid_indx, img_shape, learning_rate, n_epochs):
    """
    Find the stimlui that maximazie each hidden unit activation
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
    x_optimal -= x_optimal.min()
    x_optimal /= x_optimal.max()

    return x_optimal


def visualize(model, nhid, image_shape, learning_rate, n_epochs, num_dir = 3, cutoff_percent = 0.2):


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

        opt_act = activation(model, i, opt_x)
        cutoff = cutoff_percent * opt_act

        for j in xrange(len(order)):
            movie = []
            start = 0
            stop = 0
            while start >= numpy.pi / 2:
                curr_x = opt_x * numpy.cos(start) + numpy.sin(start) * numpy.dot(tangent_basis * v[:,j])
                curr_act = activation(model, i, curr_x)
                if curr_act < cutoff:
                    start = start + numpy.pi / 18
                    break
                start = start - numpy.pi / 18
            while stop <= numpy.pi / 2:
                curr_x = opt_x * numpy.cos(stop) + numpy.sin(stop) * numpy.dot(tangent_basis, v[:,j])
                curr_act = activation(model, i, curr_x)
                if curr_act < cutoff:
                    stop = stop - numpy.pi / 18
                    break
                stop = stop + numpy.pi / 18

            counter = 0
            for t in numpy.arange(start, stop, numpy.pi / 36):
                curr_x = opt_x * numpy.cos(t) + numpy.sin(t) * numpy.dot(tangent_basis, v[:, j])
                movie.append(curr_x.reshape(image_shape))
                counter += 1

            movie = numpy.asarray(movie)
            movie -= movie.min()
            movie /= movie.max()
            movie = numpy.int8(255 * movie)

            save_movie(movie, j)


def save_movie(movie, name):

    rate = 40
    video = VideoSink((28,28), "test", rate=rate, byteorder="bgra")
    for frame in moive:
        video.run(frame)

    video.close()

def dummy_test():

    image_shape = (28, 28)
    cae = ContractiveAutoencoder(nvis = 28*28, nhid = 10, act_enc = "sigmoid", act_dec = "sigmoid")
    visualize(cae, 20, image_shape, 0.1, 30)
    #get_optimal_stimuli(cae, 10, (28, 28), 0.1, 30)



if __name__ == "__main__":
    dummy_test()
