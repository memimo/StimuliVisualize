import subprocess

class VideoSink(object) :
    """
    credit: https://github.com/vokimon/freenect_python_processing/blob/master/src/videosink.py
    """
    def __init__( self, size, filename="output", rate=1, byteorder="bgra" ) :
        print byteorder
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

